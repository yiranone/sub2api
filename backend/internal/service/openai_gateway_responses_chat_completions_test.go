package service

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestForwardOpenAIResponsesAsChatCompletions_UsesRequestedBillingModel(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	body := []byte(`{"model":"gpt-5.4","input":"hello","stream":false}`)
	c.Request = httptest.NewRequest(http.MethodPost, "/responses", strings.NewReader(string(body)))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_resp_chat_compat"}},
		Body: io.NopCloser(strings.NewReader(`{
			"id":"chatcmpl_compat_1",
			"object":"chat.completion",
			"created":1777550000,
			"model":"doubao-seed-2-0-pro-260215",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
		}`)),
	}}

	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: upstream,
	}
	account := &Account{
		ID:          8,
		Name:        "doubao-openai-compatible",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":  "test-key",
			"base_url": "https://example.com",
			"model_mapping": map[string]any{
				"gpt-5.4": "doubao-seed-2-0-pro-260215",
			},
		},
		Extra: map[string]any{
			"openai_chat_completions_mode": true,
		},
	}

	result, err := svc.forwardOpenAIResponsesAsChatCompletions(context.Background(), c, account, body, time.Now())
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "gpt-5.4", result.Model)
	require.Equal(t, "gpt-5.4", result.BillingModel)
	require.Equal(t, "doubao-seed-2-0-pro-260215", result.UpstreamModel)
	require.Equal(t, "doubao-seed-2-0-pro-260215", gjson.GetBytes(upstream.lastBody, "model").String())
	require.Equal(t, int64(11), gjson.GetBytes(rec.Body.Bytes(), "usage.input_tokens").Int())
	require.Equal(t, int64(7), gjson.GetBytes(rec.Body.Bytes(), "usage.output_tokens").Int())
}
