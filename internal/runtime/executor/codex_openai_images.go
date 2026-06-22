package executor

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/proxyutil"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	codexOpenAIImageSourceFormat = "openai-image"
	codexImagesGenerationsPath   = "/v1/images/generations"
	codexImagesEditsPath         = "/v1/images/edits"
	codexDirectImagesGenerations = "/images/generations"
	codexDirectImagesEdits       = "/images/edits"
	codexOpenAIImagesMainModel   = "gpt-5.4-mini"

	codexImageSSEMissingCompleted = "missing_response_completed"
	codexImageSSEStreamClosed     = "upstream_stream_closed"
	codexImageSSEReadError        = "read_error"
	codexImageSSEH2Reset          = "h2_stream_reset"
	codexImageSSEContextTimeout   = "context_timeout"
	codexImageSSEContextCanceled  = "context_canceled"
	codexImageSSEUpstreamError    = "upstream_error_event"

	codexStatusClientClosedRequest = 499
)

type codexOpenAIImagePreparedRequest struct {
	Body           []byte
	ResponseFormat string
	StreamPrefix   string
	EndpointPath   string
	ContentType    string
}

type codexImageCallResult struct {
	Result        string
	RevisedPrompt string
	OutputFormat  string
	Size          string
	Background    string
	Quality       string
}

func isCodexOpenAIImageRequest(opts cliproxyexecutor.Options) bool {
	if !strings.EqualFold(strings.TrimSpace(opts.SourceFormat.String()), codexOpenAIImageSourceFormat) {
		return false
	}
	return codexIsImagesEndpointPath(helps.PayloadRequestPath(opts))
}

func codexIsImagesEndpointPath(path string) bool {
	path = strings.TrimSpace(path)
	if path == codexImagesGenerationsPath || path == codexImagesEditsPath || path == codexDirectImagesGenerations || path == codexDirectImagesEdits {
		return true
	}
	return strings.HasSuffix(path, codexDirectImagesGenerations) || strings.HasSuffix(path, codexDirectImagesEdits)
}

func codexOpenAIImageIsGenerationsPath(path string) bool {
	path = strings.TrimSpace(path)
	return path == codexImagesGenerationsPath || path == codexDirectImagesGenerations || strings.HasSuffix(path, codexDirectImagesGenerations)
}

func codexOpenAIImageIsEditsPath(path string) bool {
	path = strings.TrimSpace(path)
	return path == codexImagesEditsPath || path == codexDirectImagesEdits || strings.HasSuffix(path, codexDirectImagesEdits)
}

func (e *CodexExecutor) resolveGPTImage2BaseModel() string {
	if e == nil || e.cfg == nil {
		return codexOpenAIImagesMainModel
	}
	model := strings.TrimSpace(e.cfg.GPTImage2BaseModel)
	if model == "" {
		return codexOpenAIImagesMainModel
	}
	if strings.HasPrefix(strings.ToLower(model), "gpt-") {
		return model
	}
	return codexOpenAIImagesMainModel
}

func (e *CodexExecutor) executeOpenAIImage(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	prepared, errPrepare := codexPrepareOpenAIImageDirectRequest(req, opts, false)
	if errPrepare != nil {
		return resp, errPrepare
	}

	apiKey, baseURL := codexCreds(auth)
	if baseURL == "" {
		baseURL = "https://chatgpt.com/backend-api/codex"
	}

	baseModel := thinking.ParseSuffix(req.Model).ModelName
	reporter := helps.NewExecutorUsageReporter(ctx, e, baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)
	reporter.SetTranslatedReasoningEffort(prepared.Body, "codex")

	url := strings.TrimSuffix(baseURL, "/") + prepared.EndpointPath
	httpReq, body, identityState, errRequest := e.newCodexOpenAIImageRequest(ctx, auth, req, url, prepared.Body)
	if errRequest != nil {
		return resp, errRequest
	}
	applyCodexHeaders(httpReq, auth, apiKey, false, e.cfg)
	if prepared.ContentType != "" {
		httpReq.Header.Set("Content-Type", prepared.ContentType)
	}
	applyCodexIdentityConfuseHeaders(httpReq.Header, &identityState)
	recordCodexOpenAIImageRequest(ctx, e.cfg, e.Identifier(), auth, url, httpReq.Header.Clone(), body)

	httpClient := e.newCodexOpenAIImageHTTPClient(ctx, auth)
	httpClient = reporter.TrackHTTPClient(httpClient)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errDo)
		return resp, errDo
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("codex executor: close response body error: %v", errClose)
		}
	}()

	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	data, errRead := io.ReadAll(httpResp.Body)
	if errRead != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errRead)
		return resp, errRead
	}
	upstreamData := applyCodexIdentityConfuseResponsePayload(data, identityState)
	helps.AppendAPIResponseChunk(ctx, e.cfg, upstreamData)
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), upstreamData))
		err = newCodexStatusErr(httpResp.StatusCode, upstreamData)
		return resp, err
	}

	reporter.Publish(ctx, helps.ParseOpenAIUsage(upstreamData))
	reporter.EnsurePublished(ctx)
	clientData := applyCodexIdentityExposeResponsePayload(upstreamData, identityState)
	out, errNormalize := codexNormalizeDirectImagesResponse(clientData, prepared.ResponseFormat)
	if errNormalize != nil {
		return resp, statusErr{code: http.StatusBadGateway, msg: errNormalize.Error()}
	}
	return cliproxyexecutor.Response{Payload: out, Headers: httpResp.Header.Clone()}, nil
}

func (e *CodexExecutor) executeOpenAIImageViaResponses(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	prepared, errPrepare := codexPrepareOpenAIImageRequest(req, opts)
	if errPrepare != nil {
		return resp, errPrepare
	}

	apiKey, baseURL := codexCreds(auth)
	if baseURL == "" {
		baseURL = "https://chatgpt.com/backend-api/codex"
	}

	mainModel := e.resolveGPTImage2BaseModel()
	reporter := helps.NewExecutorUsageReporter(ctx, e, mainModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	body, errBuild := e.prepareCodexOpenAIImageBody(prepared.Body, req, opts, mainModel)
	if errBuild != nil {
		return resp, errBuild
	}
	reporter.SetTranslatedReasoningEffort(body, "codex")

	url := strings.TrimSuffix(baseURL, "/") + "/responses"
	var identityState codexIdentityConfuseState
	httpReq, body, identityState, errCache := e.cacheHelper(ctx, sdktranslator.FromString(codexOpenAIImageSourceFormat), url, auth, req, req.Payload, body)
	if errCache != nil {
		return resp, errCache
	}
	applyCodexHeaders(httpReq, auth, apiKey, true, e.cfg)
	applyCodexIdentityConfuseHeaders(httpReq.Header, &identityState)
	recordCodexOpenAIImageRequest(ctx, e.cfg, e.Identifier(), auth, url, httpReq.Header.Clone(), body)

	httpClient := e.newCodexOpenAIImageHTTPClient(ctx, auth)
	httpClient = reporter.TrackHTTPClient(httpClient)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errDo)
		return resp, errDo
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("codex executor: close response body error: %v", errClose)
		}
	}()

	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		data, errRead := io.ReadAll(httpResp.Body)
		if errRead != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, errRead)
			return resp, errRead
		}
		data = applyCodexIdentityConfuseResponsePayload(data, identityState)
		helps.AppendAPIResponseChunk(ctx, e.cfg, data)
		helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), data))
		err = newCodexStatusErr(httpResp.StatusCode, data)
		return resp, err
	}

	outputItemsByIndex := make(map[int64][]byte)
	var outputItemsFallback [][]byte
	eventData, errStream := codexReadOpenAIImageResponsesSSEWithHeaders(ctx, httpResp.Body, httpResp.Header, outputItemsByIndex, &outputItemsFallback)
	if errStream != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errStream)
		return resp, errStream
	}
	eventData = applyCodexIdentityConfuseResponsePayload(eventData, identityState)
	if detail, ok := helps.ParseCodexUsage(eventData); ok {
		reporter.Publish(ctx, detail)
	}
	publishCodexImageToolUsage(ctx, reporter, body, eventData)
	for idx, item := range outputItemsByIndex {
		outputItemsByIndex[idx] = applyCodexIdentityConfuseResponsePayload(item, identityState)
	}
	for idx, item := range outputItemsFallback {
		outputItemsFallback[idx] = applyCodexIdentityConfuseResponsePayload(item, identityState)
	}
	results, createdAt, usageRaw, firstMeta, errExtract := codexExtractImageResults(eventData, outputItemsByIndex, outputItemsFallback)
	if errExtract != nil {
		return resp, errExtract
	}
	if len(results) == 0 {
		completedData := patchCodexCompletedOutput(eventData, outputItemsByIndex, outputItemsFallback)
		reason := codexImageCompletedWithoutOutputReason(completedData)
		logCodexImageCompletedWithoutOutput(ctx, completedData, reason)
		return resp, statusErr{code: http.StatusServiceUnavailable, msg: "upstream completed without image output: " + reason}
	}
	out, errOutput := codexBuildImagesAPIResponse(results, createdAt, usageRaw, firstMeta, prepared.ResponseFormat)
	if errOutput != nil {
		return resp, errOutput
	}
	return cliproxyexecutor.Response{Payload: out, Headers: httpResp.Header.Clone()}, nil
}

type codexImageSSEStats struct {
	startedAt              time.Time
	sawResponseCompleted   bool
	sawFirstEvent          bool
	sawErrorEvent          bool
	lastEventType          string
	lastDataType           string
	eventCount             int
	dataCount              int
	imageCount             int
	partialImageCount      int
	completedOutputCount   int
	outputItemDoneCount    int
	outputItemResultCount  int
	syntheticCompletedUsed bool
	streamEndReason        string
	readErrorType          string
	upstream               codexImageSSEUpstreamErrorSummary
}

type codexImageSSEEvent struct {
	eventType string
	dataLines [][]byte
	hasEvent  bool
}

type codexImageSSEUpstreamErrorSummary struct {
	eventType        string
	dataType         string
	errorType        string
	errorCode        string
	errorStatus      string
	errorParam       string
	errorReason      string
	incompleteReason string
	failedReason     string
	upstreamResponse string
	upstreamRequest  string
	retryAfter       string
	responseID       string
	errorCategory    string
}

func codexReadOpenAIImageResponsesSSE(ctx context.Context, r io.Reader, outputItemsByIndex map[int64][]byte, outputItemsFallback *[][]byte) ([]byte, error) {
	return codexReadOpenAIImageResponsesSSEWithHeaders(ctx, r, nil, outputItemsByIndex, outputItemsFallback)
}

func (e *CodexExecutor) newCodexOpenAIImageHTTPClient(ctx context.Context, auth *cliproxyauth.Auth) *http.Client {
	var cfg *config.Config
	if e != nil {
		cfg = e.cfg
	}
	httpClient := helps.NewProxyAwareHTTPClient(ctx, cfg, auth, 0)
	if cfg == nil || !cfg.Codex.DisableHTTP2 {
		return httpClient
	}
	if httpClient.Transport == nil {
		if transport, ok := http.DefaultTransport.(*http.Transport); ok && transport != nil {
			httpClient.Transport = proxyutil.CloneTransportWithHTTP11(transport)
		} else {
			transport := &http.Transport{}
			proxyutil.DisableHTTP2ForTransport(transport)
			httpClient.Transport = transport
		}
		return httpClient
	}
	if transport, ok := httpClient.Transport.(*http.Transport); ok && transport != nil {
		httpClient.Transport = proxyutil.CloneTransportWithHTTP11(transport)
	}
	return httpClient
}

func codexReadOpenAIImageResponsesSSEWithHeaders(ctx context.Context, r io.Reader, headers http.Header, outputItemsByIndex map[int64][]byte, outputItemsFallback *[][]byte) ([]byte, error) {
	stats := &codexImageSSEStats{startedAt: time.Now()}
	stats.captureSafeHeaders(headers)
	reader := bufio.NewReader(r)
	var event codexImageSSEEvent
	syntheticCompleted := func() []byte {
		var fallback [][]byte
		if outputItemsFallback != nil {
			fallback = *outputItemsFallback
		}
		completedData := codexCompletedFromOutputItems(outputItemsByIndex, fallback)
		if len(completedData) > 0 {
			stats.syntheticCompletedUsed = true
			stats.completedOutputCount = codexImageCompletedOutputCount(completedData)
			stats.imageCount = codexImageCompletedResultCount(completedData)
		}
		return completedData
	}

	dispatch := func() ([]byte, bool, error) {
		if !event.hasPending() {
			event.reset()
			return nil, false, nil
		}
		stats.sawFirstEvent = true
		stats.eventCount++
		stats.lastEventType = codexImageSafeSummaryValue(event.eventType)

		eventData := bytes.TrimSpace(event.data())
		dataType := ""
		if len(eventData) > 0 {
			dataType = strings.TrimSpace(gjson.GetBytes(eventData, "type").String())
			stats.lastDataType = codexImageSafeSummaryValue(dataType)
		}
		if codexIsImageSSEUpstreamError(event.eventType, dataType) {
			stats.sawErrorEvent = true
			stats.streamEndReason = codexImageSSEUpstreamError
			stats.captureUpstreamError(event.eventType, dataType, eventData)
			return nil, false, stats.statusErr(codexImageSSEUpstreamError, http.StatusBadGateway)
		}
		if len(eventData) > 0 {
			switch dataType {
			case "response.output_item.done":
				stats.captureOutputItemDone(eventData)
				collectCodexOutputItemDone(eventData, outputItemsByIndex, outputItemsFallback)
			case "response.image_generation_call.partial_image":
				stats.partialImageCount++
			case "response.completed":
				stats.sawResponseCompleted = true
				stats.completedOutputCount = codexImageCompletedOutputCount(eventData)
				stats.imageCount = codexImageCompletedResultCount(eventData)
				return eventData, true, nil
			}
		}
		event.reset()
		return nil, false, nil
	}

	for {
		line, errRead := reader.ReadBytes('\n')
		if len(line) > 0 {
			codexConsumeImageSSELine(line, &event, stats)
			if codexIsBlankSSELine(line) {
				completedData, done, errDispatch := dispatch()
				if done || errDispatch != nil {
					return completedData, errDispatch
				}
			}
		}
		if errRead == nil {
			continue
		}
		if errors.Is(errRead, io.EOF) {
			completedData, done, errDispatch := dispatch()
			if done || errDispatch != nil {
				return completedData, errDispatch
			}
			if classification, code, streamEndReason, readErrorType, ok := codexClassifyImageStreamContext(ctx, nil); ok {
				stats.streamEndReason = streamEndReason
				stats.readErrorType = readErrorType
				return nil, stats.statusErr(classification, code)
			}
			if completedData := syntheticCompleted(); len(completedData) > 0 {
				return completedData, nil
			}
			stats.streamEndReason = "eof"
			return nil, stats.statusErr(codexImageSSEMissingCompleted, http.StatusServiceUnavailable)
		}

		completedData, done, errDispatch := dispatch()
		if done || errDispatch != nil {
			return completedData, errDispatch
		}
		classification, code, streamEndReason, readErrorType := codexClassifyImageStreamReadError(ctx, errRead)
		if classification != codexImageSSEContextCanceled && classification != codexImageSSEContextTimeout {
			if completedData := syntheticCompleted(); len(completedData) > 0 {
				return completedData, nil
			}
		}
		stats.streamEndReason = streamEndReason
		stats.readErrorType = readErrorType
		return nil, stats.statusErr(classification, code)
	}
}

func (s *codexImageSSEStats) captureSafeHeaders(headers http.Header) {
	if s == nil || headers == nil {
		return
	}
	s.upstream.retryAfter = codexImageSafeSummaryValue(headers.Get("Retry-After"))
	s.upstream.upstreamRequest = codexImageFirstSafeHeader(headers,
		"X-Request-Id",
		"Openai-Request-Id",
		"X-Openai-Request-Id",
		"Cf-Ray",
	)
}

func codexImageFirstSafeHeader(headers http.Header, names ...string) string {
	for _, name := range names {
		if value := codexImageSafeSummaryValue(headers.Get(name)); value != "" {
			return value
		}
	}
	return ""
}

func (s *codexImageSSEStats) captureOutputItemDone(payload []byte) {
	if s == nil || len(payload) == 0 {
		return
	}
	item := gjson.GetBytes(payload, "item")
	if item.Get("type").String() == "image_generation_call" {
		s.outputItemDoneCount++
		if strings.TrimSpace(item.Get("result").String()) != "" {
			s.outputItemResultCount++
			s.imageCount++
		}
	}
}

func codexImageCompletedOutputCount(payload []byte) int {
	output := gjson.GetBytes(payload, "response.output")
	if !output.IsArray() {
		return 0
	}
	return len(output.Array())
}

func codexImageCompletedResultCount(payload []byte) int {
	output := gjson.GetBytes(payload, "response.output")
	if !output.IsArray() {
		return 0
	}
	count := 0
	for _, item := range output.Array() {
		if item.Get("type").String() == "image_generation_call" && strings.TrimSpace(item.Get("result").String()) != "" {
			count++
		}
	}
	return count
}

func (s *codexImageSSEStats) captureUpstreamError(eventType string, dataType string, payload []byte) {
	if s == nil {
		return
	}
	summary := &s.upstream
	summary.eventType = codexImageSafeSummaryValue(eventType)
	summary.dataType = codexImageSafeSummaryValue(dataType)
	summary.errorType = codexImagePayloadFirstSafeValue(payload,
		"error.type",
		"response.error.type",
	)
	summary.errorCode = codexImagePayloadFirstSafeValue(payload,
		"error.code",
		"response.error.code",
		"code",
	)
	summary.errorStatus = codexImagePayloadFirstSafeValue(payload,
		"error.status",
		"error.status_code",
		"response.error.status",
		"response.error.status_code",
		"status",
		"response.status",
	)
	summary.errorParam = codexImagePayloadFirstSafeValue(payload,
		"error.param",
		"response.error.param",
		"param",
	)
	summary.errorReason = codexImagePayloadFirstSafeValue(payload,
		"error.reason",
		"response.error.reason",
		"reason",
	)
	summary.incompleteReason = codexImagePayloadFirstSafeValue(payload,
		"response.incomplete_details.reason",
		"incomplete_details.reason",
		"response.incomplete_reason",
		"incomplete_reason",
	)
	summary.failedReason = codexImagePayloadFirstSafeValue(payload,
		"response.failed_reason",
		"failed_reason",
		"response.status_details.reason",
	)
	if summary.failedReason == "" && strings.EqualFold(dataType, "response.failed") {
		summary.failedReason = firstNonEmpty(summary.errorReason, summary.errorCode, summary.errorType)
	}
	if summary.incompleteReason == "" && strings.EqualFold(dataType, "response.incomplete") {
		summary.incompleteReason = firstNonEmpty(summary.errorReason, summary.errorCode, summary.errorType)
	}
	summary.upstreamResponse = codexImagePayloadFirstSafeValue(payload,
		"response.id",
		"id",
	)
	summary.responseID = summary.upstreamResponse
	summary.errorCategory = codexImageClassifyUpstreamError(eventType, dataType, payload, *summary)
}

func codexImagePayloadFirstSafeValue(payload []byte, paths ...string) string {
	if len(payload) == 0 {
		return ""
	}
	for _, path := range paths {
		result := gjson.GetBytes(payload, path)
		if !result.Exists() || result.Type == gjson.Null {
			continue
		}
		if value := codexImageSafeSummaryValue(result.String()); value != "" {
			return value
		}
	}
	return ""
}

func codexImagePayloadFirstRawValue(payload []byte, paths ...string) string {
	if len(payload) == 0 {
		return ""
	}
	for _, path := range paths {
		result := gjson.GetBytes(payload, path)
		if !result.Exists() || result.Type == gjson.Null {
			continue
		}
		if value := strings.TrimSpace(result.String()); value != "" {
			return value
		}
	}
	return ""
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func codexImageClassifyUpstreamError(eventType string, dataType string, payload []byte, summary codexImageSSEUpstreamErrorSummary) string {
	status := strings.TrimSpace(summary.errorStatus)
	structuredText := strings.ToLower(strings.Join([]string{
		eventType,
		dataType,
		summary.errorType,
		summary.errorCode,
		summary.errorStatus,
		summary.errorReason,
		summary.incompleteReason,
		summary.failedReason,
	}, " "))
	messageText := strings.ToLower(codexImagePayloadFirstRawValue(payload, "error.message", "response.error.message", "message"))

	switch {
	case codexImageContainsAny(structuredText, "quota", "insufficient_quota", "usage_limit", "usage limit"):
		return "quota"
	case codexImageContainsAny(structuredText, "capacity"):
		return "capacity"
	case codexImageContainsAny(structuredText, "overloaded", "overload"):
		return "overloaded"
	case codexImageContainsAny(structuredText, "rate_limit", "rate limit", "too_many_requests", "too many requests") || status == "429":
		return "rate_limit"
	case codexImageContainsAny(structuredText, "safety", "policy", "content_filter", "content filter"):
		return "safety"
	case codexImageContainsAny(structuredText, "timeout", "deadline"):
		return "timeout"
	case codexImageContainsAny(structuredText, "unauthorized", "forbidden", "authentication", "auth") || status == "401" || status == "403":
		return "auth"
	case codexImageContainsAny(structuredText, "invalid_request", "invalid request", "bad_request", "bad request", "unprocessable", "malformed") || status == "400" || status == "422":
		return "invalid_request"
	case strings.EqualFold(dataType, "response.failed"):
		return "upstream_failed"
	case strings.EqualFold(dataType, "response.incomplete"):
		return "upstream_incomplete"
	case codexImageContainsAny(structuredText, "internal_error", "internal error", "server_error", "server error") || strings.HasPrefix(status, "5"):
		return "internal_error"
	case codexImageContainsAny(messageText, "quota", "insufficient_quota", "usage_limit", "usage limit"):
		return "quota"
	case codexImageContainsAny(messageText, "capacity"):
		return "capacity"
	case codexImageContainsAny(messageText, "overloaded", "overload"):
		return "overloaded"
	case codexImageContainsAny(messageText, "rate_limit", "rate limit", "too_many_requests", "too many requests"):
		return "rate_limit"
	case codexImageContainsAny(messageText, "safety", "policy", "content_filter", "content filter"):
		return "safety"
	case codexImageContainsAny(messageText, "timeout", "deadline"):
		return "timeout"
	case codexImageContainsAny(messageText, "unauthorized", "forbidden", "authentication"):
		return "auth"
	case codexImageContainsAny(messageText, "invalid_request", "invalid request", "bad_request", "bad request", "unprocessable", "malformed"):
		return "invalid_request"
	case codexImageContainsAny(messageText, "internal_error", "internal error", "server_error", "server error"):
		return "internal_error"
	default:
		return "unknown"
	}
}

func codexImageContainsAny(value string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
}

func codexConsumeImageSSELine(line []byte, event *codexImageSSEEvent, stats *codexImageSSEStats) {
	line = bytes.TrimRight(line, "\r\n")
	if len(line) == 0 || bytes.HasPrefix(line, []byte(":")) {
		return
	}

	field, value := codexSplitSSEField(line)
	switch field {
	case "event":
		event.eventType = strings.TrimSpace(string(value))
		event.hasEvent = true
	case "data":
		stats.dataCount++
		copied := make([]byte, len(value))
		copy(copied, value)
		event.dataLines = append(event.dataLines, copied)
	case "id", "retry":
		return
	default:
		return
	}
}

func codexSplitSSEField(line []byte) (string, []byte) {
	fieldBytes, value, ok := bytes.Cut(line, []byte(":"))
	if !ok {
		return string(line), nil
	}
	if len(value) > 0 && value[0] == ' ' {
		value = value[1:]
	}
	return string(fieldBytes), value
}

func codexIsBlankSSELine(line []byte) bool {
	return len(bytes.TrimSpace(line)) == 0
}

func (e *codexImageSSEEvent) hasPending() bool {
	return e != nil && (e.hasEvent || len(e.dataLines) > 0)
}

func (e *codexImageSSEEvent) data() []byte {
	if e == nil || len(e.dataLines) == 0 {
		return nil
	}
	return bytes.Join(e.dataLines, []byte("\n"))
}

func (e *codexImageSSEEvent) reset() {
	if e == nil {
		return
	}
	e.eventType = ""
	e.dataLines = nil
	e.hasEvent = false
}

func codexIsImageSSEUpstreamError(eventType string, dataType string) bool {
	if strings.EqualFold(strings.TrimSpace(eventType), "error") {
		return true
	}
	switch dataType {
	case "error", "response.failed", "response.incomplete":
		return true
	default:
		return false
	}
}

func codexClassifyImageStreamReadError(ctx context.Context, err error) (classification string, code int, streamEndReason string, readErrorType string) {
	if classification, code, streamEndReason, readErrorType, ok := codexClassifyImageStreamContext(ctx, err); ok {
		return classification, code, streamEndReason, readErrorType
	}
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return codexImageSSEStreamClosed, http.StatusServiceUnavailable, "unexpected_eof", codexImageSSEStreamClosed
	}
	if codexIsHTTP2StreamResetError(err) {
		return codexImageSSEH2Reset, http.StatusBadGateway, "read_error", codexImageSSEH2Reset
	}
	return codexImageSSEReadError, http.StatusServiceUnavailable, "read_error", codexImageSSEReadError
}

func codexClassifyImageStreamContext(ctx context.Context, err error) (classification string, code int, streamEndReason string, readErrorType string, ok bool) {
	if ctx != nil {
		ctxErr := ctx.Err()
		if errors.Is(ctxErr, context.DeadlineExceeded) {
			return codexImageSSEContextTimeout, http.StatusGatewayTimeout, codexImageSSEContextTimeout, "context_deadline_exceeded", true
		}
		if errors.Is(ctxErr, context.Canceled) {
			return codexImageSSEContextCanceled, codexStatusClientClosedRequest, codexImageSSEContextCanceled, "context_canceled", true
		}
	}
	if errors.Is(err, context.DeadlineExceeded) || codexIsTimeoutError(err) {
		return codexImageSSEContextTimeout, http.StatusGatewayTimeout, codexImageSSEContextTimeout, "context_deadline_exceeded", true
	}
	if errors.Is(err, context.Canceled) {
		return codexImageSSEContextCanceled, codexStatusClientClosedRequest, codexImageSSEContextCanceled, "context_canceled", true
	}
	return "", 0, "", "", false
}

func codexIsTimeoutError(err error) bool {
	if err == nil {
		return false
	}
	var timeout interface{ Timeout() bool }
	if errors.As(err, &timeout) && timeout.Timeout() {
		return true
	}
	lower := strings.ToLower(err.Error())
	return strings.Contains(lower, "context deadline exceeded") ||
		strings.Contains(lower, "deadline exceeded") ||
		strings.Contains(lower, "timeout")
}

func codexIsHTTP2StreamResetError(err error) bool {
	if err == nil {
		return false
	}
	lower := strings.ToLower(err.Error())
	return strings.Contains(lower, "stream id") ||
		strings.Contains(lower, "internal_error") ||
		strings.Contains(lower, "received from peer") ||
		strings.Contains(lower, "http2") ||
		strings.Contains(lower, "rst_stream")
}

func (s *codexImageSSEStats) statusErr(classification string, code int) statusErr {
	if s == nil {
		return statusErr{code: code, msg: fmt.Sprintf("codex image stream error: classification=%s", classification)}
	}
	elapsedMS := int64(0)
	if !s.startedAt.IsZero() {
		elapsedMS = time.Since(s.startedAt).Milliseconds()
	}
	return statusErr{
		code: code,
		msg: fmt.Sprintf(
			"codex image stream error: classification=%s saw_response_completed=%t saw_first_event=%t saw_error_event=%t last_event_type=%s last_data_type=%s upstream_event_type=%s upstream_data_type=%s upstream_error_type=%s upstream_error_code=%s upstream_error_status=%s upstream_error_param=%s upstream_error_reason=%s upstream_incomplete_reason=%s upstream_failed_reason=%s upstream_response_id=%s upstream_request_id=%s retry_after=%s response_id=%s error_category=%s event_count=%d data_count=%d image_count=%d partial_image_count=%d completed_seen=%t completed_output_count=%d output_item_done_count=%d output_item_result_count=%d synthetic_completed_used=%t elapsed_ms=%d stream_end_reason=%s read_error_type=%s",
			classification,
			s.sawResponseCompleted,
			s.sawFirstEvent,
			s.sawErrorEvent,
			codexImageSafeSummaryValue(s.lastEventType),
			codexImageSafeSummaryValue(s.lastDataType),
			codexImageSafeSummaryValue(s.upstream.eventType),
			codexImageSafeSummaryValue(s.upstream.dataType),
			codexImageSafeSummaryValue(s.upstream.errorType),
			codexImageSafeSummaryValue(s.upstream.errorCode),
			codexImageSafeSummaryValue(s.upstream.errorStatus),
			codexImageSafeSummaryValue(s.upstream.errorParam),
			codexImageSafeSummaryValue(s.upstream.errorReason),
			codexImageSafeSummaryValue(s.upstream.incompleteReason),
			codexImageSafeSummaryValue(s.upstream.failedReason),
			codexImageSafeSummaryValue(s.upstream.upstreamResponse),
			codexImageSafeSummaryValue(s.upstream.upstreamRequest),
			codexImageSafeSummaryValue(s.upstream.retryAfter),
			codexImageSafeSummaryValue(s.upstream.responseID),
			codexImageSafeSummaryValue(s.upstream.errorCategory),
			s.eventCount,
			s.dataCount,
			s.imageCount,
			s.partialImageCount,
			s.sawResponseCompleted,
			s.completedOutputCount,
			s.outputItemDoneCount,
			s.outputItemResultCount,
			s.syntheticCompletedUsed,
			elapsedMS,
			codexImageSafeSummaryValue(s.streamEndReason),
			codexImageSafeSummaryValue(s.readErrorType),
		),
	}
}

func codexImageSafeSummaryValue(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	lower := strings.ToLower(value)
	for _, sensitive := range []string{"authorization", "cookie", "api_key", "access_token", "refresh_token", "id_token", "bearer", "sk-", "prompt", "base64", "b64"} {
		if strings.Contains(lower, sensitive) {
			return "redacted"
		}
	}
	if len(value) > 80 {
		value = value[:80]
	}
	var b strings.Builder
	b.Grow(len(value))
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '.' || r == '_' || r == '-' || r == '/' || r == ':':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}

func codexImageCompletedWithoutOutputReason(payload []byte) string {
	for _, path := range []string{
		"response.error.code",
		"response.error.type",
		"response.error.message",
		"response.incomplete_details.reason",
		"response.incomplete_details.type",
	} {
		if value := strings.TrimSpace(gjson.GetBytes(payload, path).String()); value != "" {
			return sanitizeCodexImageOutputReason(value)
		}
	}

	output := gjson.GetBytes(payload, "response.output")
	if output.IsArray() {
		items := output.Array()
		if len(items) == 0 {
			return "empty_response_output"
		}
		for _, item := range items {
			if item.Get("type").String() != "image_generation_call" {
				continue
			}
			if status := strings.TrimSpace(item.Get("status").String()); status != "" {
				return sanitizeCodexImageOutputReason(status)
			}
			if errCode := strings.TrimSpace(item.Get("error.code").String()); errCode != "" {
				return sanitizeCodexImageOutputReason(errCode)
			}
			if errMessage := strings.TrimSpace(item.Get("error.message").String()); errMessage != "" {
				return sanitizeCodexImageOutputReason(errMessage)
			}
			return "empty_image_generation_result"
		}
	}
	if status := strings.TrimSpace(gjson.GetBytes(payload, "response.status").String()); status != "" {
		return sanitizeCodexImageOutputReason(status)
	}
	return "empty_response_output"
}

func sanitizeCodexImageOutputReason(reason string) string {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		return "unknown"
	}
	replacer := strings.NewReplacer("\r", " ", "\n", " ", "\t", " ")
	reason = replacer.Replace(reason)
	reason = strings.Join(strings.Fields(reason), " ")
	if len(reason) > 120 {
		reason = strings.TrimSpace(reason[:120])
	}
	return reason
}

func codexImageOutputTypes(payload []byte) []string {
	output := gjson.GetBytes(payload, "response.output")
	if !output.IsArray() {
		return nil
	}
	types := make([]string, 0)
	for _, item := range output.Array() {
		if itemType := strings.TrimSpace(item.Get("type").String()); itemType != "" {
			types = append(types, itemType)
		}
	}
	return types
}

func codexImageGenerationStatuses(payload []byte) []string {
	output := gjson.GetBytes(payload, "response.output")
	if !output.IsArray() {
		return nil
	}
	statuses := make([]string, 0)
	for _, item := range output.Array() {
		if item.Get("type").String() != "image_generation_call" {
			continue
		}
		if status := strings.TrimSpace(item.Get("status").String()); status != "" {
			statuses = append(statuses, status)
		}
	}
	return statuses
}

func logCodexImageCompletedWithoutOutput(ctx context.Context, payload []byte, reason string) {
	fields := log.Fields{
		"reason":                    reason,
		"completed_seen":            gjson.GetBytes(payload, "type").String() == "response.completed",
		"completed_output_count":    codexImageCompletedOutputCount(payload),
		"output_item_result_count":  codexImageCompletedResultCount(payload),
		"response_status":           strings.TrimSpace(gjson.GetBytes(payload, "response.status").String()),
		"response_error":            strings.TrimSpace(gjson.GetBytes(payload, "response.error").Raw),
		"response_incomplete":       strings.TrimSpace(gjson.GetBytes(payload, "response.incomplete_details").Raw),
		"response_output_types":     codexImageOutputTypes(payload),
		"image_generation_statuses": codexImageGenerationStatuses(payload),
	}
	helps.LogWithRequestID(ctx).WithFields(fields).Warn("codex openai images: upstream completed without image output")
}

func (e *CodexExecutor) executeOpenAIImageStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	prepared, errPrepare := codexPrepareOpenAIImageDirectRequest(req, opts, true)
	if errPrepare != nil {
		return nil, errPrepare
	}

	apiKey, baseURL := codexCreds(auth)
	if baseURL == "" {
		baseURL = "https://chatgpt.com/backend-api/codex"
	}

	baseModel := thinking.ParseSuffix(req.Model).ModelName
	reporter := helps.NewExecutorUsageReporter(ctx, e, baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)
	reporter.SetTranslatedReasoningEffort(prepared.Body, "codex")

	url := strings.TrimSuffix(baseURL, "/") + prepared.EndpointPath
	httpReq, body, identityState, errRequest := e.newCodexOpenAIImageRequest(ctx, auth, req, url, prepared.Body)
	if errRequest != nil {
		return nil, errRequest
	}
	applyCodexHeaders(httpReq, auth, apiKey, true, e.cfg)
	if prepared.ContentType != "" {
		httpReq.Header.Set("Content-Type", prepared.ContentType)
	}
	applyCodexIdentityConfuseHeaders(httpReq.Header, &identityState)
	recordCodexOpenAIImageRequest(ctx, e.cfg, e.Identifier(), auth, url, httpReq.Header.Clone(), body)

	httpClient := e.newCodexOpenAIImageHTTPClient(ctx, auth)
	httpClient = reporter.TrackHTTPClient(httpClient)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errDo)
		return nil, errDo
	}
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		data, errRead := io.ReadAll(httpResp.Body)
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("codex executor: close response body error: %v", errClose)
		}
		if errRead != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, errRead)
			return nil, errRead
		}
		data = applyCodexIdentityConfuseResponsePayload(data, identityState)
		helps.AppendAPIResponseChunk(ctx, e.cfg, data)
		helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), data))
		err = newCodexStatusErr(httpResp.StatusCode, data)
		return nil, err
	}

	out := make(chan cliproxyexecutor.StreamChunk)
	go func() {
		defer close(out)
		defer reporter.EnsurePublished(ctx)
		defer func() {
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("codex executor: close response body error: %v", errClose)
			}
		}()

		buffer := make([]byte, 32*1024)
		for {
			n, errRead := httpResp.Body.Read(buffer)
			if n > 0 {
				upstreamChunk := applyCodexIdentityConfuseResponsePayload(bytes.Clone(buffer[:n]), identityState)
				helps.AppendAPIResponseChunk(ctx, e.cfg, upstreamChunk)
				clientChunk := applyCodexIdentityExposeResponsePayload(upstreamChunk, identityState)
				select {
				case out <- cliproxyexecutor.StreamChunk{Payload: clientChunk}:
				case <-ctx.Done():
					return
				}
			}
			if errRead != nil {
				if errRead != io.EOF {
					helps.RecordAPIResponseError(ctx, e.cfg, errRead)
					reporter.PublishFailure(ctx, errRead)
					select {
					case out <- cliproxyexecutor.StreamChunk{Err: errRead}:
					case <-ctx.Done():
					}
				}
				return
			}
		}
	}()
	return &cliproxyexecutor.StreamResult{Headers: httpResp.Header.Clone(), Chunks: out}, nil
}

func (e *CodexExecutor) executeOpenAIImageResponsesStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	prepared, errPrepare := codexPrepareOpenAIImageRequest(req, opts)
	if errPrepare != nil {
		return nil, errPrepare
	}

	apiKey, baseURL := codexCreds(auth)
	if baseURL == "" {
		baseURL = "https://chatgpt.com/backend-api/codex"
	}

	mainModel := e.resolveGPTImage2BaseModel()
	reporter := helps.NewExecutorUsageReporter(ctx, e, mainModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	body, errBuild := e.prepareCodexOpenAIImageBody(prepared.Body, req, opts, mainModel)
	if errBuild != nil {
		return nil, errBuild
	}
	reporter.SetTranslatedReasoningEffort(body, "codex")

	url := strings.TrimSuffix(baseURL, "/") + "/responses"
	var identityState codexIdentityConfuseState
	httpReq, body, identityState, errCache := e.cacheHelper(ctx, sdktranslator.FromString(codexOpenAIImageSourceFormat), url, auth, req, req.Payload, body)
	if errCache != nil {
		return nil, errCache
	}
	applyCodexHeaders(httpReq, auth, apiKey, true, e.cfg)
	applyCodexIdentityConfuseHeaders(httpReq.Header, &identityState)
	recordCodexOpenAIImageRequest(ctx, e.cfg, e.Identifier(), auth, url, httpReq.Header.Clone(), body)

	httpClient := e.newCodexOpenAIImageHTTPClient(ctx, auth)
	httpClient = reporter.TrackHTTPClient(httpClient)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errDo)
		return nil, errDo
	}
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		data, errRead := io.ReadAll(httpResp.Body)
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("codex executor: close response body error: %v", errClose)
		}
		if errRead != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, errRead)
			return nil, errRead
		}
		data = applyCodexIdentityConfuseResponsePayload(data, identityState)
		helps.AppendAPIResponseChunk(ctx, e.cfg, data)
		helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), data))
		err = newCodexStatusErr(httpResp.StatusCode, data)
		return nil, err
	}

	out := make(chan cliproxyexecutor.StreamChunk)
	go func() {
		defer close(out)
		defer func() {
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("codex executor: close response body error: %v", errClose)
			}
		}()

		sendPayload := func(payload []byte) bool {
			select {
			case out <- cliproxyexecutor.StreamChunk{Payload: payload}:
				return true
			case <-ctx.Done():
				return false
			}
		}
		sendError := func(errSend error) bool {
			select {
			case out <- cliproxyexecutor.StreamChunk{Err: errSend}:
				return true
			case <-ctx.Done():
				return false
			}
		}

		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(nil, 52_428_800)
		outputItemsByIndex := make(map[int64][]byte)
		var outputItemsFallback [][]byte
		emitCompletedData := func(completedData []byte, itemsByIndex map[int64][]byte, fallback [][]byte) bool {
			if detail, ok := helps.ParseCodexUsage(completedData); ok {
				reporter.Publish(ctx, detail)
			}
			publishCodexImageToolUsage(ctx, reporter, body, completedData)
			results, _, usageRaw, _, errExtract := codexExtractImageResults(completedData, itemsByIndex, fallback)
			if errExtract != nil {
				sendError(errExtract)
				return true
			}
			if len(results) == 0 {
				return false
			}
			for _, img := range results {
				frame := codexBuildImageCompletedFrame(img, usageRaw, prepared.ResponseFormat, prepared.StreamPrefix)
				if len(frame) > 0 && !sendPayload(frame) {
					return true
				}
			}
			return true
		}
		emitSyntheticCompleted := func() bool {
			completedData := codexCompletedFromOutputItems(outputItemsByIndex, outputItemsFallback)
			if len(completedData) == 0 {
				return false
			}
			return emitCompletedData(completedData, nil, nil)
		}
		for scanner.Scan() {
			line := applyCodexIdentityConfuseResponsePayload(scanner.Bytes(), identityState)
			helps.AppendAPIResponseChunk(ctx, e.cfg, line)
			if !bytes.HasPrefix(line, dataTag) {
				continue
			}
			eventData := bytes.TrimSpace(line[len(dataTag):])
			switch gjson.GetBytes(eventData, "type").String() {
			case "response.output_item.done":
				collectCodexOutputItemDone(eventData, outputItemsByIndex, &outputItemsFallback)
			case "response.image_generation_call.partial_image":
				frame := codexBuildImagePartialFrame(eventData, prepared.ResponseFormat, prepared.StreamPrefix)
				if len(frame) > 0 && !sendPayload(frame) {
					return
				}
			case "response.completed":
				if emitCompletedData(eventData, outputItemsByIndex, outputItemsFallback) {
					return
				}
				completedData := patchCodexCompletedOutput(eventData, outputItemsByIndex, outputItemsFallback)
				reason := codexImageCompletedWithoutOutputReason(completedData)
				logCodexImageCompletedWithoutOutput(ctx, completedData, reason)
				sendError(statusErr{code: http.StatusServiceUnavailable, msg: "upstream completed without image output: " + reason})
				return
			}
		}
		if errScan := scanner.Err(); errScan != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, errScan)
			reporter.PublishFailure(ctx, errScan)
			if emitSyntheticCompleted() {
				return
			}
			sendError(errScan)
			return
		}
		if !emitSyntheticCompleted() && len(outputItemsByIndex)+len(outputItemsFallback) > 0 {
			completedData := patchCodexCompletedOutput([]byte(`{"type":"response.completed","response":{"output":[]}}`), outputItemsByIndex, outputItemsFallback)
			if len(completedData) > 0 {
				results, _, _, _, _ := codexExtractImageResults(completedData, nil, nil)
				if len(results) == 0 {
					reason := codexImageCompletedWithoutOutputReason(completedData)
					logCodexImageCompletedWithoutOutput(ctx, completedData, reason)
					sendError(statusErr{code: http.StatusServiceUnavailable, msg: "upstream completed without image output: " + reason})
				}
			}
		}
	}()
	return &cliproxyexecutor.StreamResult{Headers: httpResp.Header.Clone(), Chunks: out}, nil
}

func (e *CodexExecutor) prepareCodexOpenAIImageBody(body []byte, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, mainModel string) ([]byte, error) {
	out := body
	mainModel = strings.TrimSpace(mainModel)
	if mainModel == "" {
		mainModel = codexOpenAIImagesMainModel
	}
	var errThinking error
	out, errThinking = thinking.ApplyThinking(out, mainModel, codexOpenAIImageSourceFormat, "codex", e.Identifier())
	if errThinking != nil {
		return nil, errThinking
	}

	requestedModel := helps.PayloadRequestedModel(opts, req.Model)
	requestPath := helps.PayloadRequestPath(opts)
	out = helps.ApplyPayloadConfigWithRequest(e.cfg, mainModel, "codex", codexOpenAIImageSourceFormat, "", out, body, requestedModel, requestPath, opts.Headers)
	out, _ = sjson.SetBytes(out, "model", mainModel)
	out, _ = sjson.SetBytes(out, "stream", true)
	out, _ = sjson.DeleteBytes(out, "previous_response_id")
	out, _ = sjson.DeleteBytes(out, "prompt_cache_retention")
	out, _ = sjson.DeleteBytes(out, "safety_identifier")
	out, _ = sjson.DeleteBytes(out, "stream_options")
	out = codexEnsureImagesResponsesImageGenerationTool(out, codexOpenAIImageToolModel("", requestedModel), codexImageActionFromRequestPath(requestPath))
	return normalizeCodexInstructions(out), nil
}

func recordCodexOpenAIImageRequest(ctx context.Context, cfg *config.Config, provider string, auth *cliproxyauth.Auth, url string, headers http.Header, body []byte) {
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, cfg, helps.UpstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   headers,
		Body:      body,
		Provider:  provider,
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})
}

func (e *CodexExecutor) newCodexOpenAIImageRequest(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, url string, body []byte) (*http.Request, []byte, codexIdentityConfuseState, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	out := body
	var identityState codexIdentityConfuseState
	if json.Valid(out) {
		out, identityState = applyCodexIdentityConfuseBody(e.cfg, auth, req.Payload, out)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(out))
	if err != nil {
		return nil, nil, codexIdentityConfuseState{}, err
	}
	return httpReq, out, identityState, nil
}

func codexPrepareOpenAIImageDirectRequest(req cliproxyexecutor.Request, opts cliproxyexecutor.Options, stream bool) (codexOpenAIImagePreparedRequest, error) {
	endpointPath, streamPrefix, errPath := codexOpenAIImageDirectEndpointPath(helps.PayloadRequestPath(opts))
	if errPath != nil {
		return codexOpenAIImagePreparedRequest{}, errPath
	}

	contentType := codexImageContentType(opts.Headers)
	if endpointPath == codexDirectImagesGenerations {
		return codexPrepareOpenAIImageDirectJSON(req.Payload, req.Model, endpointPath, streamPrefix, stream)
	}

	mediaType, _, _ := mime.ParseMediaType(contentType)
	if strings.HasPrefix(strings.ToLower(mediaType), "multipart/") {
		return codexPrepareOpenAIImageDirectMultipart(req.Payload, req.Model, endpointPath, streamPrefix, contentType, stream)
	}
	return codexPrepareOpenAIImageDirectJSON(req.Payload, req.Model, endpointPath, streamPrefix, stream)
}

func codexOpenAIImageDirectEndpointPath(path string) (endpointPath string, streamPrefix string, err error) {
	path = strings.TrimSpace(path)
	if strings.HasSuffix(path, codexDirectImagesEdits) {
		return codexDirectImagesEdits, "image_edit", nil
	}
	if strings.HasSuffix(path, codexDirectImagesGenerations) {
		return codexDirectImagesGenerations, "image_generation", nil
	}
	return "", "", fmt.Errorf("unsupported OpenAI image endpoint path %q", path)
}

func codexPrepareOpenAIImageDirectJSON(rawJSON []byte, routeModel string, endpointPath string, streamPrefix string, stream bool) (codexOpenAIImagePreparedRequest, error) {
	if !json.Valid(rawJSON) {
		return codexOpenAIImagePreparedRequest{}, fmt.Errorf("invalid OpenAI image request JSON")
	}
	payload := bytes.Clone(rawJSON)
	payload, _ = sjson.SetBytes(payload, "model", codexOpenAIImageDirectModel(routeModel))
	if stream {
		payload, _ = sjson.SetBytes(payload, "stream", true)
	} else {
		payload, _ = sjson.DeleteBytes(payload, "stream")
	}
	return codexOpenAIImagePreparedRequest{
		Body:           payload,
		ResponseFormat: codexOpenAIImageResponseFormatFromJSON(rawJSON),
		StreamPrefix:   streamPrefix,
		EndpointPath:   endpointPath,
		ContentType:    "application/json",
	}, nil
}

func codexPrepareOpenAIImageDirectMultipart(rawBody []byte, routeModel string, endpointPath string, streamPrefix string, contentType string, stream bool) (codexOpenAIImagePreparedRequest, error) {
	_, params, errMedia := mime.ParseMediaType(contentType)
	if errMedia != nil {
		return codexOpenAIImagePreparedRequest{}, fmt.Errorf("parse multipart content type failed: %w", errMedia)
	}
	boundary := strings.TrimSpace(params["boundary"])
	if boundary == "" {
		return codexOpenAIImagePreparedRequest{}, fmt.Errorf("multipart boundary is required")
	}
	reader := multipart.NewReader(bytes.NewReader(rawBody), boundary)
	form, errForm := reader.ReadForm(32 << 20)
	if errForm != nil {
		return codexOpenAIImagePreparedRequest{}, fmt.Errorf("parse multipart form failed: %w", errForm)
	}
	defer func() {
		if errRemove := form.RemoveAll(); errRemove != nil {
			log.Errorf("codex openai images: remove multipart temp files error: %v", errRemove)
		}
	}()

	body, errRewrite := codexBuildOpenAIImageDirectMultipartJSONPayload(form, codexOpenAIImageDirectModel(routeModel), stream)
	if errRewrite != nil {
		return codexOpenAIImagePreparedRequest{}, errRewrite
	}
	return codexOpenAIImagePreparedRequest{
		Body:           body,
		ResponseFormat: codexNormalizeImageResponseFormat(codexFormValue(form, "response_format")),
		StreamPrefix:   streamPrefix,
		EndpointPath:   endpointPath,
		ContentType:    "application/json",
	}, nil
}

func codexOpenAIImageDirectModel(routeModel string) string {
	model := strings.TrimSpace(thinking.ParseSuffix(routeModel).ModelName)
	if idx := strings.LastIndex(model, "/"); idx >= 0 && idx < len(model)-1 {
		model = strings.TrimSpace(model[idx+1:])
	}
	if model == "" {
		return codexDefaultImageToolModel
	}
	return model
}

func codexBuildOpenAIImageDirectMultipartJSONPayload(form *multipart.Form, model string, stream bool) ([]byte, error) {
	if form == nil {
		return nil, fmt.Errorf("multipart form is nil")
	}
	payload := make(map[string]any)
	if strings.TrimSpace(model) != "" {
		payload["model"] = strings.TrimSpace(model)
	}
	if stream {
		payload["stream"] = true
	}

	for key, values := range form.Value {
		if key == "model" || key == "stream" || len(values) == 0 {
			continue
		}
		if len(values) == 1 {
			payload[key] = codexMultipartJSONFieldValue(key, values[0])
			continue
		}
		out := make([]any, 0, len(values))
		for _, value := range values {
			out = append(out, codexMultipartJSONFieldValue(key, value))
		}
		payload[key] = out
	}

	images := make([]map[string]string, 0)
	for _, fh := range codexMultipartImageFiles(form) {
		dataURL, errData := codexMultipartFileToDataURL(fh)
		if errData != nil {
			return nil, errData
		}
		images = append(images, map[string]string{"image_url": dataURL})
	}
	if len(images) > 0 {
		payload["images"] = images
	}

	if maskFiles := form.File["mask"]; len(maskFiles) > 0 && maskFiles[0] != nil {
		dataURL, errData := codexMultipartFileToDataURL(maskFiles[0])
		if errData != nil {
			return nil, errData
		}
		payload["mask"] = map[string]string{"image_url": dataURL}
	}

	body, errMarshal := json.Marshal(payload)
	if errMarshal != nil {
		return nil, fmt.Errorf("marshal image edit JSON failed: %w", errMarshal)
	}
	return body, nil
}

func codexMultipartJSONFieldValue(key string, value string) any {
	value = strings.TrimSpace(value)
	switch key {
	case "n", "output_compression", "partial_images":
		if parsed, errParse := strconv.ParseInt(value, 10, 64); errParse == nil {
			return parsed
		}
	}
	return value
}

func codexPrepareOpenAIImageRequest(req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (codexOpenAIImagePreparedRequest, error) {
	path := helps.PayloadRequestPath(opts)
	if codexOpenAIImageIsGenerationsPath(path) {
		return codexPrepareOpenAIImageGenerationJSON(req.Payload, req.Model)
	}
	if !codexOpenAIImageIsEditsPath(path) {
		return codexOpenAIImagePreparedRequest{}, fmt.Errorf("unsupported OpenAI image endpoint path %q", path)
	}

	contentType := codexImageContentType(opts.Headers)
	mediaType, _, _ := mime.ParseMediaType(contentType)
	if strings.HasPrefix(strings.ToLower(mediaType), "multipart/") {
		return codexPrepareOpenAIImageEditMultipart(req.Payload, req.Model, contentType)
	}
	return codexPrepareOpenAIImageEditJSON(req.Payload, req.Model)
}

func codexPrepareOpenAIImageGenerationJSON(rawJSON []byte, routeModel string) (codexOpenAIImagePreparedRequest, error) {
	if !json.Valid(rawJSON) {
		return codexOpenAIImagePreparedRequest{}, fmt.Errorf("invalid OpenAI image generation request JSON")
	}
	prompt := strings.TrimSpace(gjson.GetBytes(rawJSON, "prompt").String())
	tool := codexBuildOpenAIImageTool(rawJSON, routeModel, "generate", []string{"size", "quality", "background", "output_format", "moderation"}, []string{"output_compression", "partial_images"})
	body := codexBuildImagesResponsesRequest(prompt, nil, tool)
	return codexOpenAIImagePreparedRequest{
		Body:           body,
		ResponseFormat: codexOpenAIImageResponseFormatFromJSON(rawJSON),
		StreamPrefix:   "image_generation",
	}, nil
}

func codexPrepareOpenAIImageEditJSON(rawJSON []byte, routeModel string) (codexOpenAIImagePreparedRequest, error) {
	if !json.Valid(rawJSON) {
		return codexOpenAIImagePreparedRequest{}, fmt.Errorf("invalid OpenAI image edit request JSON")
	}
	prompt := strings.TrimSpace(gjson.GetBytes(rawJSON, "prompt").String())
	images := make([]string, 0)
	if imagesResult := gjson.GetBytes(rawJSON, "images"); imagesResult.IsArray() {
		for _, img := range imagesResult.Array() {
			url := strings.TrimSpace(img.Get("image_url").String())
			if url != "" {
				images = append(images, url)
			}
		}
	}
	tool := codexBuildOpenAIImageTool(rawJSON, routeModel, "edit", []string{"size", "quality", "background", "output_format", "input_fidelity", "moderation"}, []string{"output_compression", "partial_images"})
	if mask := strings.TrimSpace(gjson.GetBytes(rawJSON, "mask.image_url").String()); mask != "" {
		tool, _ = sjson.SetBytes(tool, "input_image_mask.image_url", mask)
	}
	body := codexBuildImagesResponsesRequest(prompt, images, tool)
	return codexOpenAIImagePreparedRequest{
		Body:           body,
		ResponseFormat: codexOpenAIImageResponseFormatFromJSON(rawJSON),
		StreamPrefix:   "image_edit",
	}, nil
}

func codexPrepareOpenAIImageEditMultipart(rawBody []byte, routeModel string, contentType string) (codexOpenAIImagePreparedRequest, error) {
	_, params, errMedia := mime.ParseMediaType(contentType)
	if errMedia != nil {
		return codexOpenAIImagePreparedRequest{}, fmt.Errorf("parse multipart content type failed: %w", errMedia)
	}
	boundary := strings.TrimSpace(params["boundary"])
	if boundary == "" {
		return codexOpenAIImagePreparedRequest{}, fmt.Errorf("multipart boundary is required")
	}
	reader := multipart.NewReader(bytes.NewReader(rawBody), boundary)
	form, errForm := reader.ReadForm(32 << 20)
	if errForm != nil {
		return codexOpenAIImagePreparedRequest{}, fmt.Errorf("parse multipart form failed: %w", errForm)
	}
	defer func() {
		if errRemove := form.RemoveAll(); errRemove != nil {
			log.Errorf("codex openai images: remove multipart temp files error: %v", errRemove)
		}
	}()

	prompt := strings.TrimSpace(codexFormValue(form, "prompt"))
	responseFormat := codexNormalizeImageResponseFormat(codexFormValue(form, "response_format"))
	tool := []byte(`{"type":"image_generation","action":"edit"}`)
	tool, _ = sjson.SetBytes(tool, "model", codexOpenAIImageToolModel(codexFormValue(form, "model"), routeModel))
	for _, field := range []string{"size", "quality", "background", "output_format", "input_fidelity", "moderation"} {
		if value := strings.TrimSpace(codexFormValue(form, field)); value != "" {
			tool, _ = sjson.SetBytes(tool, field, value)
		}
	}
	for _, field := range []string{"output_compression", "partial_images"} {
		if value := strings.TrimSpace(codexFormValue(form, field)); value != "" {
			if parsed, errParse := strconv.ParseInt(value, 10, 64); errParse == nil {
				tool, _ = sjson.SetBytes(tool, field, parsed)
			}
		}
	}

	images := make([]string, 0)
	for _, fh := range codexMultipartImageFiles(form) {
		dataURL, errData := codexMultipartFileToDataURL(fh)
		if errData != nil {
			return codexOpenAIImagePreparedRequest{}, errData
		}
		images = append(images, dataURL)
	}
	if maskFiles := form.File["mask"]; len(maskFiles) > 0 && maskFiles[0] != nil {
		dataURL, errData := codexMultipartFileToDataURL(maskFiles[0])
		if errData != nil {
			return codexOpenAIImagePreparedRequest{}, errData
		}
		tool, _ = sjson.SetBytes(tool, "input_image_mask.image_url", dataURL)
	}

	body := codexBuildImagesResponsesRequest(prompt, images, tool)
	return codexOpenAIImagePreparedRequest{
		Body:           body,
		ResponseFormat: responseFormat,
		StreamPrefix:   "image_edit",
	}, nil
}

func codexImageContentType(headers http.Header) string {
	if headers == nil {
		return ""
	}
	return strings.TrimSpace(headers.Get("Content-Type"))
}

func codexOpenAIImageResponseFormatFromJSON(rawJSON []byte) string {
	return codexNormalizeImageResponseFormat(gjson.GetBytes(rawJSON, "response_format").String())
}

func codexNormalizeImageResponseFormat(responseFormat string) string {
	if strings.EqualFold(strings.TrimSpace(responseFormat), "url") {
		return "url"
	}
	return "b64_json"
}

func codexOpenAIImageToolModel(requestModel string, routeModel string) string {
	model := strings.TrimSpace(requestModel)
	if model == "" {
		model = strings.TrimSpace(routeModel)
	}
	if idx := strings.LastIndex(model, "/"); idx >= 0 && idx < len(model)-1 {
		model = strings.TrimSpace(model[idx+1:])
	}
	if model == "" {
		model = codexDefaultImageToolModel
	}
	return model
}

func codexBuildOpenAIImageTool(rawJSON []byte, routeModel string, action string, stringFields []string, numberFields []string) []byte {
	tool := []byte(`{"type":"image_generation","action":""}`)
	tool, _ = sjson.SetBytes(tool, "action", action)
	tool, _ = sjson.SetBytes(tool, "model", codexOpenAIImageToolModel(gjson.GetBytes(rawJSON, "model").String(), routeModel))
	for _, field := range stringFields {
		if value := strings.TrimSpace(gjson.GetBytes(rawJSON, field).String()); value != "" {
			tool, _ = sjson.SetBytes(tool, field, value)
		}
	}
	for _, field := range numberFields {
		if value := gjson.GetBytes(rawJSON, field); value.Exists() && value.Type == gjson.Number {
			tool, _ = sjson.SetBytes(tool, field, value.Int())
		}
	}
	return tool
}

func codexBuildImagesResponsesRequest(prompt string, images []string, toolJSON []byte) []byte {
	req := []byte(`{"instructions":"","stream":true,"reasoning":{"effort":"medium","summary":"auto"},"parallel_tool_calls":true,"include":["reasoning.encrypted_content"],"model":"","store":false,"tool_choice":{"type":"image_generation"}}`)
	req, _ = sjson.SetBytes(req, "model", codexOpenAIImagesMainModel)

	input := []byte(`[{"type":"message","role":"user","content":[{"type":"input_text","text":""}]}]`)
	input, _ = sjson.SetBytes(input, "0.content.0.text", prompt)
	contentIndex := 1
	for _, img := range images {
		if strings.TrimSpace(img) == "" {
			continue
		}
		part := []byte(`{"type":"input_image","image_url":""}`)
		part, _ = sjson.SetBytes(part, "image_url", img)
		input, _ = sjson.SetRawBytes(input, fmt.Sprintf("0.content.%d", contentIndex), part)
		contentIndex++
	}
	req, _ = sjson.SetRawBytes(req, "input", input)

	req, _ = sjson.SetRawBytes(req, "tools", []byte(`[]`))
	if len(toolJSON) > 0 && json.Valid(toolJSON) {
		req, _ = sjson.SetRawBytes(req, "tools.-1", toolJSON)
	}
	return codexEnsureImagesResponsesImageGenerationTool(req, codexDefaultImageToolModel, "")
}

func codexEnsureImagesResponsesImageGenerationTool(req []byte, fallbackModel string, action string) []byte {
	if len(req) == 0 || !codexImageToolChoiceForcesImageGeneration(req) || codexImagesResponsesHasImageGenerationTool(req) {
		return req
	}
	tools := gjson.GetBytes(req, "tools")
	if !tools.IsArray() {
		req, _ = sjson.SetRawBytes(req, "tools", []byte(`[]`))
	}
	tool := []byte(`{"type":"image_generation"}`)
	if model := strings.TrimSpace(fallbackModel); model != "" {
		tool, _ = sjson.SetBytes(tool, "model", model)
	}
	if action = strings.TrimSpace(action); action != "" {
		tool, _ = sjson.SetBytes(tool, "action", action)
	}
	req, _ = sjson.SetRawBytes(req, "tools.-1", tool)
	return req
}

func codexImageToolChoiceForcesImageGeneration(req []byte) bool {
	choice := gjson.GetBytes(req, "tool_choice")
	if !choice.Exists() {
		return false
	}
	if choice.Type == gjson.String {
		return strings.EqualFold(strings.TrimSpace(choice.String()), "image_generation")
	}
	choiceType := strings.TrimSpace(choice.Get("type").String())
	if strings.EqualFold(choiceType, "image_generation") {
		return true
	}
	return strings.EqualFold(choiceType, "tool") && strings.EqualFold(strings.TrimSpace(choice.Get("name").String()), "image_generation")
}

func codexImagesResponsesHasImageGenerationTool(req []byte) bool {
	tools := gjson.GetBytes(req, "tools")
	if !tools.IsArray() {
		return false
	}
	for _, tool := range tools.Array() {
		if strings.EqualFold(strings.TrimSpace(tool.Get("type").String()), "image_generation") {
			return true
		}
	}
	return false
}

func codexImageActionFromRequestPath(requestPath string) string {
	if codexOpenAIImageIsEditsPath(requestPath) {
		return "edit"
	}
	if codexOpenAIImageIsGenerationsPath(requestPath) {
		return "generate"
	}
	return ""
}

func codexFormValue(form *multipart.Form, key string) string {
	if form == nil || len(form.Value[key]) == 0 {
		return ""
	}
	return strings.TrimSpace(form.Value[key][0])
}

func codexMultipartImageFiles(form *multipart.Form) []*multipart.FileHeader {
	if form == nil {
		return nil
	}
	if files := form.File["image[]"]; len(files) > 0 {
		return files
	}
	return form.File["image"]
}

func codexMultipartFileToDataURL(fileHeader *multipart.FileHeader) (string, error) {
	if fileHeader == nil {
		return "", fmt.Errorf("upload file is nil")
	}
	f, errOpen := fileHeader.Open()
	if errOpen != nil {
		return "", fmt.Errorf("open upload file failed: %w", errOpen)
	}
	defer func() {
		if errClose := f.Close(); errClose != nil {
			log.Errorf("codex openai images: close upload file error: %v", errClose)
		}
	}()

	data, errRead := io.ReadAll(f)
	if errRead != nil {
		return "", fmt.Errorf("read upload file failed: %w", errRead)
	}
	mediaType := strings.TrimSpace(fileHeader.Header.Get("Content-Type"))
	if mediaType == "" {
		mediaType = http.DetectContentType(data)
	}
	return "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(data), nil
}

type codexDirectImageResult struct {
	B64JSON       string
	URL           string
	RevisedPrompt string
	MimeType      string
	OutputFormat  string
	Size          string
	Background    string
	Quality       string
}

func codexNormalizeDirectImagesResponse(payload []byte, responseFormat string) ([]byte, error) {
	if !json.Valid(payload) {
		return nil, fmt.Errorf("upstream returned invalid image response JSON")
	}

	root := gjson.ParseBytes(payload)
	createdAt := root.Get("created").Int()
	if createdAt <= 0 {
		createdAt = root.Get("created_at").Int()
	}
	if createdAt <= 0 {
		createdAt = time.Now().Unix()
	}

	results := make([]codexDirectImageResult, 0)
	appendResult := func(item gjson.Result) {
		result := codexDirectImageResult{
			B64JSON:       strings.TrimSpace(item.Get("b64_json").String()),
			URL:           strings.TrimSpace(item.Get("url").String()),
			RevisedPrompt: strings.TrimSpace(item.Get("revised_prompt").String()),
			MimeType:      strings.TrimSpace(item.Get("mime_type").String()),
			OutputFormat:  strings.TrimSpace(item.Get("output_format").String()),
			Size:          strings.TrimSpace(item.Get("size").String()),
			Background:    strings.TrimSpace(item.Get("background").String()),
			Quality:       strings.TrimSpace(item.Get("quality").String()),
		}
		if result.OutputFormat == "" && strings.Contains(result.MimeType, "/") {
			result.OutputFormat = strings.TrimPrefix(result.MimeType, "image/")
		}
		if result.MimeType == "" {
			result.MimeType = codexMimeTypeFromOutputFormat(result.OutputFormat)
		}
		if result.B64JSON == "" && strings.HasPrefix(result.URL, "data:") {
			if comma := strings.Index(result.URL, ","); comma >= 0 && comma < len(result.URL)-1 {
				result.B64JSON = strings.TrimSpace(result.URL[comma+1:])
			}
		}
		if result.B64JSON == "" && result.URL == "" {
			return
		}
		results = append(results, result)
	}

	if data := root.Get("data"); data.IsArray() {
		for _, item := range data.Array() {
			appendResult(item)
		}
	} else if eventType := strings.TrimSpace(root.Get("type").String()); eventType == "image_generation.completed" || eventType == "image_edit.completed" {
		appendResult(root)
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("upstream did not return image output")
	}

	out := []byte(`{"created":0,"data":[]}`)
	out, _ = sjson.SetBytes(out, "created", createdAt)
	responseFormat = codexNormalizeImageResponseFormat(responseFormat)
	for _, img := range results {
		item := []byte(`{}`)
		if responseFormat == "url" {
			if img.URL != "" {
				item, _ = sjson.SetBytes(item, "url", img.URL)
			} else {
				item, _ = sjson.SetBytes(item, "url", "data:"+img.MimeType+";base64,"+img.B64JSON)
			}
		} else if img.B64JSON != "" {
			item, _ = sjson.SetBytes(item, "b64_json", img.B64JSON)
		} else {
			item, _ = sjson.SetBytes(item, "url", img.URL)
		}
		if img.RevisedPrompt != "" {
			item, _ = sjson.SetBytes(item, "revised_prompt", img.RevisedPrompt)
		}
		out, _ = sjson.SetRawBytes(out, "data.-1", item)
	}

	first := results[0]
	for _, field := range []struct {
		name     string
		fallback string
	}{
		{name: "background", fallback: first.Background},
		{name: "output_format", fallback: first.OutputFormat},
		{name: "quality", fallback: first.Quality},
		{name: "size", fallback: first.Size},
	} {
		if value := strings.TrimSpace(root.Get(field.name).String()); value != "" {
			out, _ = sjson.SetBytes(out, field.name, value)
			continue
		}
		if field.fallback != "" {
			out, _ = sjson.SetBytes(out, field.name, field.fallback)
		}
	}

	if usage := root.Get("usage"); usage.Exists() && usage.IsObject() {
		out, _ = sjson.SetRawBytes(out, "usage", []byte(usage.Raw))
	}
	return out, nil
}

// codexExtractImageResults extracts image generation results directly from the
// completed event and the items collected from response.output_item.done events,
// without rebuilding the full completed JSON.
//
// It prefers image_generation_call items already present in the completed event's
// response.output and only falls back to the collected items when that output is
// empty, mirroring the semantics of patchCodexCompletedOutput + the previous
// extractor. Skipping the concatenate-and-reparse step avoids two large copies of
// the base64 payload, which matters for multi-megabyte generated images.
func codexExtractImageResults(completed []byte, itemsByIndex map[int64][]byte, fallback [][]byte) (results []codexImageCallResult, createdAt int64, usageRaw []byte, firstMeta codexImageCallResult, err error) {
	if gjson.GetBytes(completed, "type").String() != "response.completed" {
		return nil, 0, nil, codexImageCallResult{}, fmt.Errorf("unexpected event type")
	}
	createdAt = gjson.GetBytes(completed, "response.created_at").Int()
	if createdAt <= 0 {
		createdAt = time.Now().Unix()
	}

	appendItem := func(item gjson.Result) {
		if item.Get("type").String() != "image_generation_call" {
			return
		}
		res := strings.TrimSpace(item.Get("result").String())
		if res == "" {
			return
		}
		entry := codexImageCallResult{
			Result:        res,
			RevisedPrompt: strings.TrimSpace(item.Get("revised_prompt").String()),
			OutputFormat:  strings.TrimSpace(item.Get("output_format").String()),
			Size:          strings.TrimSpace(item.Get("size").String()),
			Background:    strings.TrimSpace(item.Get("background").String()),
			Quality:       strings.TrimSpace(item.Get("quality").String()),
		}
		if len(results) == 0 {
			firstMeta = entry
		}
		results = append(results, entry)
	}

	var outputItems []gjson.Result
	if output := gjson.GetBytes(completed, "response.output"); output.Exists() && output.IsArray() {
		outputItems = output.Array()
	}
	if len(outputItems) > 0 {
		// Completed event already carries the output; extract from it in place.
		results = make([]codexImageCallResult, 0, len(outputItems))
		for _, item := range outputItems {
			appendItem(item)
		}
	} else if len(itemsByIndex) > 0 || len(fallback) > 0 {
		// Completed output was empty; extract directly from the collected items,
		// preserving their original output_index ordering.
		results = make([]codexImageCallResult, 0, len(itemsByIndex)+len(fallback))
		if len(itemsByIndex) > 0 {
			indexes := make([]int64, 0, len(itemsByIndex))
			for idx := range itemsByIndex {
				indexes = append(indexes, idx)
			}
			sort.Slice(indexes, func(i, j int) bool { return indexes[i] < indexes[j] })
			for _, idx := range indexes {
				appendItem(gjson.ParseBytes(itemsByIndex[idx]))
			}
		}
		for _, raw := range fallback {
			appendItem(gjson.ParseBytes(raw))
		}
	}

	if usage := gjson.GetBytes(completed, "response.tool_usage.image_gen"); usage.Exists() && usage.IsObject() {
		usageRaw = []byte(usage.Raw)
	}
	return results, createdAt, usageRaw, firstMeta, nil
}

func codexBuildImagesAPIResponse(results []codexImageCallResult, createdAt int64, usageRaw []byte, firstMeta codexImageCallResult, responseFormat string) ([]byte, error) {
	out := []byte(`{"created":0,"data":[]}`)
	out, _ = sjson.SetBytes(out, "created", createdAt)
	responseFormat = codexNormalizeImageResponseFormat(responseFormat)
	for _, img := range results {
		item := []byte(`{}`)
		if responseFormat == "url" {
			item, _ = sjson.SetBytes(item, "url", "data:"+codexMimeTypeFromOutputFormat(img.OutputFormat)+";base64,"+img.Result)
		} else {
			item, _ = sjson.SetBytes(item, "b64_json", img.Result)
		}
		if img.RevisedPrompt != "" {
			item, _ = sjson.SetBytes(item, "revised_prompt", img.RevisedPrompt)
		}
		out, _ = sjson.SetRawBytes(out, "data.-1", item)
	}
	if firstMeta.Background != "" {
		out, _ = sjson.SetBytes(out, "background", firstMeta.Background)
	}
	if firstMeta.OutputFormat != "" {
		out, _ = sjson.SetBytes(out, "output_format", firstMeta.OutputFormat)
	}
	if firstMeta.Quality != "" {
		out, _ = sjson.SetBytes(out, "quality", firstMeta.Quality)
	}
	if firstMeta.Size != "" {
		out, _ = sjson.SetBytes(out, "size", firstMeta.Size)
	}
	if len(usageRaw) > 0 && json.Valid(usageRaw) {
		out, _ = sjson.SetRawBytes(out, "usage", usageRaw)
	}
	return out, nil
}

func codexBuildImagePartialFrame(payload []byte, responseFormat string, streamPrefix string) []byte {
	b64 := strings.TrimSpace(gjson.GetBytes(payload, "partial_image_b64").String())
	if b64 == "" {
		return nil
	}
	outputFormat := strings.TrimSpace(gjson.GetBytes(payload, "output_format").String())
	eventName := strings.TrimSpace(streamPrefix) + ".partial_image"
	data := []byte(`{"type":"","partial_image_index":0}`)
	data, _ = sjson.SetBytes(data, "type", eventName)
	data, _ = sjson.SetBytes(data, "partial_image_index", gjson.GetBytes(payload, "partial_image_index").Int())
	if codexNormalizeImageResponseFormat(responseFormat) == "url" {
		data, _ = sjson.SetBytes(data, "url", "data:"+codexMimeTypeFromOutputFormat(outputFormat)+";base64,"+b64)
	} else {
		data, _ = sjson.SetBytes(data, "b64_json", b64)
	}
	return codexBuildSSEFrame(eventName, data)
}

func codexBuildImageCompletedFrame(img codexImageCallResult, usageRaw []byte, responseFormat string, streamPrefix string) []byte {
	eventName := strings.TrimSpace(streamPrefix) + ".completed"
	data := []byte(`{"type":""}`)
	data, _ = sjson.SetBytes(data, "type", eventName)
	if codexNormalizeImageResponseFormat(responseFormat) == "url" {
		data, _ = sjson.SetBytes(data, "url", "data:"+codexMimeTypeFromOutputFormat(img.OutputFormat)+";base64,"+img.Result)
	} else {
		data, _ = sjson.SetBytes(data, "b64_json", img.Result)
	}
	if len(usageRaw) > 0 && json.Valid(usageRaw) {
		data, _ = sjson.SetRawBytes(data, "usage", usageRaw)
	}
	if img.RevisedPrompt != "" {
		data, _ = sjson.SetBytes(data, "revised_prompt", img.RevisedPrompt)
	}
	if img.OutputFormat != "" {
		data, _ = sjson.SetBytes(data, "output_format", img.OutputFormat)
	}
	if img.Background != "" {
		data, _ = sjson.SetBytes(data, "background", img.Background)
	}
	if img.Quality != "" {
		data, _ = sjson.SetBytes(data, "quality", img.Quality)
	}
	if img.Size != "" {
		data, _ = sjson.SetBytes(data, "size", img.Size)
	}
	return codexBuildSSEFrame(eventName, data)
}

func codexBuildSSEFrame(eventName string, data []byte) []byte {
	var buf bytes.Buffer
	if strings.TrimSpace(eventName) != "" {
		buf.WriteString("event: ")
		buf.WriteString(eventName)
		buf.WriteString("\n")
	}
	buf.WriteString("data: ")
	buf.Write(data)
	buf.WriteString("\n\n")
	return buf.Bytes()
}

func codexMimeTypeFromOutputFormat(outputFormat string) string {
	switch strings.ToLower(strings.TrimSpace(outputFormat)) {
	case "jpg", "jpeg":
		return "image/jpeg"
	case "webp":
		return "image/webp"
	default:
		return "image/png"
	}
}
