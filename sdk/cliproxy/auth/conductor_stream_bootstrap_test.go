package auth

import (
	"net/http"
	"testing"
)

func TestStreamBootstrapErrorPreservesEmptyStreamServiceUnavailable(t *testing.T) {
	emptyErr := &Error{
		Code:       "empty_stream",
		Message:    "upstream stream closed before first payload",
		Retryable:  true,
		HTTPStatus: http.StatusServiceUnavailable,
	}

	wrapped := newStreamBootstrapError(emptyErr, nil)
	status, ok := wrapped.(interface{ StatusCode() int })
	if !ok {
		t.Fatalf("wrapped error does not expose StatusCode(): %T", wrapped)
	}
	if got := status.StatusCode(); got != http.StatusServiceUnavailable {
		t.Fatalf("StatusCode() = %d, want %d", got, http.StatusServiceUnavailable)
	}
	if !emptyErr.Retryable {
		t.Fatal("empty stream error should remain retryable")
	}
}
