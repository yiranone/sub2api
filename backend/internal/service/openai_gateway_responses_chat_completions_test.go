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

func TestMinimizeOpenAICompatChatRequestForProvider_MiniMaxDropsUnsupportedFields(t *testing.T) {
	account := &Account{
		ID:       8,
		Platform: PlatformOpenAI,
		Type:     AccountTypeAPIKey,
		Credentials: map[string]any{
			"api_key":  "sk-minimax-test",
			"base_url": "https://api.minimax.chat",
		},
	}
	body := []byte(`{
		"model":"minimax-m2.7",
		"messages":[{"role":"user","content":"hello"}],
		"stream":true,
		"stream_options":{"include_usage":true},
		"reasoning_effort":"high",
		"service_tier":"priority",
		"temperature":1,
		"top_p":1,
		"tools":[{"type":"function","function":{"name":"x"}}],
		"tool_choice":"auto",
		"instructions":"abc"
	}`)

	minimized, changed := minimizeOpenAICompatChatRequestForProvider(account, body)
	require.True(t, changed)
	require.Equal(t, "codex-MiniMax-M2.7", gjson.GetBytes(minimized, "model").String())
	require.True(t, gjson.GetBytes(minimized, "stream").Bool())
	require.Equal(t, "hello", gjson.GetBytes(minimized, "messages.0.content").String())
	require.False(t, gjson.GetBytes(minimized, "stream_options").Exists())
	require.False(t, gjson.GetBytes(minimized, "reasoning_effort").Exists())
	require.False(t, gjson.GetBytes(minimized, "service_tier").Exists())
	require.False(t, gjson.GetBytes(minimized, "temperature").Exists())
	require.False(t, gjson.GetBytes(minimized, "top_p").Exists())
	require.False(t, gjson.GetBytes(minimized, "tools").Exists())
	require.False(t, gjson.GetBytes(minimized, "tool_choice").Exists())
	require.False(t, gjson.GetBytes(minimized, "instructions").Exists())
}

func TestMinimizeOpenAICompatChatRequestForProvider_MiniMaxFoldsSystemIntoUser(t *testing.T) {
	account := &Account{
		ID:       8,
		Platform: PlatformOpenAI,
		Type:     AccountTypeAPIKey,
		Credentials: map[string]any{
			"api_key":  "sk-minimax-test",
			"base_url": "https://api.minimax.chat",
		},
	}
	body := []byte(`{
		"model":"MiniMax-M2.7",
		"messages":[
			{"role":"system","content":"system rules"},
			{"role":"developer","content":"developer rules"},
			{"role":"user","content":"hello"}
		],
		"stream":true,
		"temperature":1
	}`)

	minimized, changed := minimizeOpenAICompatChatRequestForProvider(account, body)
	require.True(t, changed)
	require.Equal(t, int64(1), gjson.GetBytes(minimized, "messages.#").Int())
	require.Equal(t, "user", gjson.GetBytes(minimized, "messages.0.role").String())
	require.Contains(t, gjson.GetBytes(minimized, "messages.0.content").String(), "system rules")
	require.Contains(t, gjson.GetBytes(minimized, "messages.0.content").String(), "developer rules")
	require.Contains(t, gjson.GetBytes(minimized, "messages.0.content").String(), "hello")
}
