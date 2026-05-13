package service

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestOpenAIGatewayServiceForwardVideos_APIKeyMiniMaxMappedModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"gpt-video-1","prompt":"make a short video","duration":6,"resolution":"768P","prompt_optimizer":true}`)

	req := httptest.NewRequest(http.MethodPost, "/v1/videos/generations", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = req

	svc := &OpenAIGatewayService{cfg: &config.Config{}}
	parsed, err := svc.ParseOpenAIVideosRequest(c, body)
	require.NoError(t, err)

	upstream := &httpUpstreamRecorder{
		responses: []*http.Response{
			{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid-create"}},
				Body:       io.NopCloser(strings.NewReader(`{"task_id":"task-123"}`)),
			},
			{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"status":"Success","file_id":"file-456"}`)),
			},
			{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"file":{"download_url":"https://cdn.example.com/video.mp4"}}`)),
			},
		},
	}
	svc.httpUpstream = upstream
	account := &Account{
		ID:          8,
		Name:        "minimax-video",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":  "test-api-key",
			"base_url": "https://api.minimaxi.com",
			"model_mapping": map[string]any{
				"gpt-video-1": "MiniMax-Hailuo-2.3",
			},
		},
	}

	result, err := svc.ForwardVideos(context.Background(), c, account, body, parsed, "")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "gpt-video-1", result.Model)
	require.Equal(t, "MiniMax-Hailuo-2.3", result.UpstreamModel)
	require.Len(t, upstream.requests, 3)
	require.Equal(t, "https://api.minimaxi.com/v1/video_generation", upstream.requests[0].URL.String())
	require.Equal(t, "https://api.minimaxi.com/v1/query/video_generation?task_id=task-123", upstream.requests[1].URL.String())
	require.Equal(t, "https://api.minimaxi.com/v1/files/retrieve?file_id=file-456", upstream.requests[2].URL.String())
	require.Equal(t, "MiniMax-Hailuo-2.3", gjson.GetBytes(upstream.bodies[0], "model").String())
	require.Equal(t, "make a short video", gjson.GetBytes(upstream.bodies[0], "prompt").String())
	require.True(t, gjson.GetBytes(upstream.bodies[0], "prompt_optimizer").Bool())
	require.Equal(t, "https://cdn.example.com/video.mp4", gjson.GetBytes(rec.Body.Bytes(), "data.0.url").String())
	require.Equal(t, "task-123", gjson.GetBytes(rec.Body.Bytes(), "data.0.task_id").String())
	require.Equal(t, "file-456", gjson.GetBytes(rec.Body.Bytes(), "data.0.file_id").String())
}

func TestBuildMiniMaxVideoURLNormalizesConfiguredEndpoint(t *testing.T) {
	require.Equal(t, "https://api.minimaxi.com/v1/video_generation", buildMiniMaxVideoGenerationURL("https://api.minimaxi.com"))
	require.Equal(t, "https://api.minimaxi.com/v1/video_generation", buildMiniMaxVideoGenerationURL("https://api.minimaxi.com/anthropic"))
	require.Equal(t, "https://api.minimaxi.com/v1/video_generation", buildMiniMaxVideoGenerationURL("https://api.minimaxi.com/anthropic/v1/messages"))
	require.Equal(t, "https://api.minimaxi.com/v1/video_generation", buildMiniMaxVideoGenerationURL("https://api.minimaxi.com/v1/chat/completions"))
	require.Equal(t, "https://api.minimaxi.com/v1/video_generation", buildMiniMaxVideoGenerationURL("https://api.minimaxi.com/v1/text/chatcompletion_v2"))
	require.Equal(t, "https://api.minimaxi.com/v1/video_generation", buildMiniMaxVideoGenerationURL("https://api.minimaxi.com/v1/video_generation"))
	require.Equal(t, "https://api.minimaxi.com/v1/query/video_generation?task_id=task-123", buildMiniMaxVideoQueryURL("https://api.minimaxi.com/v1/chat/completions", "task-123"))
	require.Equal(t, "https://api.minimaxi.com/v1/files/retrieve?file_id=file-456", buildMiniMaxVideoFileRetrieveURL("https://api.minimaxi.com/v1/chat/completions", "file-456"))
}

func TestOpenAIGatewayServiceForwardVideos_MiniMaxBaseRespError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"gpt-video-1","prompt":"make a short video"}`)

	req := httptest.NewRequest(http.MethodPost, "/v1/videos/generations", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = req

	svc := &OpenAIGatewayService{cfg: &config.Config{}}
	parsed, err := svc.ParseOpenAIVideosRequest(c, body)
	require.NoError(t, err)

	upstream := &httpUpstreamRecorder{
		responses: []*http.Response{
			{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"base_resp":{"status_code":1004,"status_msg":"invalid model"}}`)),
			},
		},
	}
	svc.httpUpstream = upstream
	account := &Account{
		ID:          9,
		Name:        "minimax-video",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":  "test-api-key",
			"base_url": "https://api.minimaxi.com",
			"model_mapping": map[string]any{
				"gpt-video-1": "MiniMax-Hailuo-2.3",
			},
		},
	}

	result, err := svc.ForwardVideos(context.Background(), c, account, body, parsed, "")
	require.Nil(t, result)
	require.ErrorContains(t, err, "MiniMax API error: status_code=1004 status_msg=invalid model")
}

func TestParseOpenAIVideosRequestRejectsNonVideoModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"gpt-image-1","prompt":"make a short video"}`)

	req := httptest.NewRequest(http.MethodPost, "/v1/videos/generations", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = req

	svc := &OpenAIGatewayService{}
	parsed, err := svc.ParseOpenAIVideosRequest(c, body)
	require.Nil(t, parsed)
	require.ErrorContains(t, err, `videos endpoint requires a video model`)
}
