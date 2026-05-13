package service

import (
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

func TestAccountTestService_WriteFullSongUsesLyricsEndpoint(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/admin/accounts/1/test", nil)

	upstream := &httpUpstreamRecorder{
		resp: &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"lyrics":"line one\nline two","base_resp":{"status_code":0,"status_msg":"success"}}`)),
		},
	}
	svc := &AccountTestService{httpUpstream: upstream, cfg: &config.Config{}}
	account := &Account{
		ID:          61,
		Name:        "minimax-lyrics",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":  "test-api-key",
			"base_url": "https://api.minimaxi.com",
			"model_mapping": map[string]any{
				"write_full_song": "write_full_song",
			},
		},
	}

	err := svc.testOpenAIAccountConnection(c, account, "write_full_song", "write lyrics about rain", "")
	require.NoError(t, err)
	require.NotNil(t, upstream.lastReq)
	require.Equal(t, "https://api.minimaxi.com/v1/lyrics_generation", upstream.lastReq.URL.String())
	require.Equal(t, "write_full_song", gjson.GetBytes(upstream.lastBody, "mode").String())
	require.Equal(t, "write lyrics about rain", gjson.GetBytes(upstream.lastBody, "prompt").String())
	require.Contains(t, rec.Body.String(), "line one\\nline two")
	require.Contains(t, rec.Body.String(), "\"success\":true")
}

func TestAccountTestService_MusicTestEmitsGeneratedLyrics(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/v1/admin/accounts/1/test", nil)

	upstream := &httpUpstreamRecorder{
		responses: []*http.Response{
			{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"lyrics":"generated line one\ngenerated line two","base_resp":{"status_code":0,"status_msg":"success"}}`)),
			},
			{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"data":{"audio":"https://cdn.example.com/music.mp3"},"extra_info":{"music_duration":5000},"base_resp":{"status_code":0,"status_msg":"success"}}`)),
			},
			{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"audio/mpeg"}},
				Body:       io.NopCloser(strings.NewReader("music-bytes")),
			},
		},
	}
	svc := &AccountTestService{httpUpstream: upstream, cfg: &config.Config{}}
	account := &Account{
		ID:          62,
		Name:        "minimax-music",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":  "test-api-key",
			"base_url": "https://api.minimaxi.com",
			"model_mapping": map[string]any{
				"gpt-music-1": "music-2.6",
			},
		},
	}

	err := svc.testOpenAIAccountConnection(c, account, "gpt-music-1", "short piano theme", "")
	require.NoError(t, err)
	require.Len(t, upstream.requests, 3)
	require.Equal(t, "https://api.minimaxi.com/v1/lyrics_generation", upstream.requests[0].URL.String())
	require.Equal(t, "https://api.minimaxi.com/v1/music_generation", upstream.requests[1].URL.String())
	require.Equal(t, "https://cdn.example.com/music.mp3", upstream.requests[2].URL.String())
	require.Equal(t, "generated line one\ngenerated line two", gjson.GetBytes(upstream.bodies[1], "lyrics").String())
	require.Contains(t, rec.Body.String(), "Generated lyrics:")
	require.Contains(t, rec.Body.String(), "generated line one\\ngenerated line two")
	require.Contains(t, rec.Body.String(), "\"audio_url\":\"data:audio/mpeg;base64,bXVzaWMtYnl0ZXM=\"")
}
