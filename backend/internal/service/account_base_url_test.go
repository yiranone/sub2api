//go:build unit

package service

import (
	"testing"
)

func TestGetBaseURL(t *testing.T) {
	tests := []struct {
		name     string
		account  Account
		expected string
	}{
		{
			name: "non-apikey type returns empty",
			account: Account{
				Type:     AccountTypeOAuth,
				Platform: PlatformAnthropic,
			},
			expected: "",
		},
		{
			name: "apikey without base_url returns default anthropic",
			account: Account{
				Type:        AccountTypeAPIKey,
				Platform:    PlatformAnthropic,
				Credentials: map[string]any{},
			},
			expected: "https://api.anthropic.com",
		},
		{
			name: "apikey with custom base_url",
			account: Account{
				Type:        AccountTypeAPIKey,
				Platform:    PlatformAnthropic,
				Credentials: map[string]any{"base_url": "https://custom.example.com"},
			},
			expected: "https://custom.example.com",
		},
		{
			name: "anthropic minimax base_url uses root host",
			account: Account{
				Type:        AccountTypeAPIKey,
				Platform:    PlatformAnthropic,
				Credentials: map[string]any{"base_url": "https://api.minimaxi.com"},
			},
			expected: "https://api.minimaxi.com",
		},
		{
			name: "anthropic minimax base_url strips explicit anthropic path",
			account: Account{
				Type:        AccountTypeAPIKey,
				Platform:    PlatformAnthropic,
				Credentials: map[string]any{"base_url": "https://api.minimaxi.com/anthropic/"},
			},
			expected: "https://api.minimaxi.com",
		},
		{
			name: "antigravity apikey auto-appends /antigravity",
			account: Account{
				Type:        AccountTypeAPIKey,
				Platform:    PlatformAntigravity,
				Credentials: map[string]any{"base_url": "https://upstream.example.com"},
			},
			expected: "https://upstream.example.com/antigravity",
		},
		{
			name: "antigravity apikey trims trailing slash before appending",
			account: Account{
				Type:        AccountTypeAPIKey,
				Platform:    PlatformAntigravity,
				Credentials: map[string]any{"base_url": "https://upstream.example.com/"},
			},
			expected: "https://upstream.example.com/antigravity",
		},
		{
			name: "antigravity non-apikey returns empty",
			account: Account{
				Type:        AccountTypeOAuth,
				Platform:    PlatformAntigravity,
				Credentials: map[string]any{"base_url": "https://upstream.example.com"},
			},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.account.GetBaseURL()
			if result != tt.expected {
				t.Errorf("GetBaseURL() = %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestGetAnthropicBaseURLForModel(t *testing.T) {
	account := Account{
		Type:        AccountTypeAPIKey,
		Platform:    PlatformAnthropic,
		Credentials: map[string]any{"base_url": "https://api.minimaxi.com"},
	}
	if got := account.GetAnthropicBaseURLForModel("claude-3-5-sonnet-latest"); got != "https://api.minimaxi.com/anthropic" {
		t.Errorf("GetAnthropicBaseURLForModel(claude) = %q", got)
	}
	if got := account.GetAnthropicBaseURLForModel("MiniMax-M2.7"); got != "https://api.minimaxi.com/anthropic" {
		t.Errorf("GetAnthropicBaseURLForModel(MiniMax-M2.7) = %q", got)
	}
	if got := account.GetAnthropicBaseURLForModel("music-2.6"); got != "https://api.minimaxi.com" {
		t.Errorf("GetAnthropicBaseURLForModel(music-2.6) = %q", got)
	}

	account.Credentials["base_url"] = "https://api.minimaxi.com/anthropic"
	if got := account.GetAnthropicBaseURLForModel("claude-3-5-sonnet-latest"); got != "https://api.minimaxi.com/anthropic" {
		t.Errorf("GetAnthropicBaseURLForModel(claude explicit anthropic) = %q", got)
	}
	if got := account.GetAnthropicBaseURLForModel("MiniMax-M2.7"); got != "https://api.minimaxi.com/anthropic" {
		t.Errorf("GetAnthropicBaseURLForModel(MiniMax-M2.7 explicit anthropic) = %q", got)
	}
	if got := account.GetAnthropicBaseURLForModel("music-2.6"); got != "https://api.minimaxi.com" {
		t.Errorf("GetAnthropicBaseURLForModel(music-2.6 explicit anthropic) = %q", got)
	}
}

func TestGetGeminiBaseURL(t *testing.T) {
	const defaultGeminiURL = "https://generativelanguage.googleapis.com"

	tests := []struct {
		name     string
		account  Account
		expected string
	}{
		{
			name: "apikey without base_url returns default",
			account: Account{
				Type:        AccountTypeAPIKey,
				Platform:    PlatformGemini,
				Credentials: map[string]any{},
			},
			expected: defaultGeminiURL,
		},
		{
			name: "apikey with custom base_url",
			account: Account{
				Type:        AccountTypeAPIKey,
				Platform:    PlatformGemini,
				Credentials: map[string]any{"base_url": "https://custom-gemini.example.com"},
			},
			expected: "https://custom-gemini.example.com",
		},
		{
			name: "antigravity apikey auto-appends /antigravity",
			account: Account{
				Type:        AccountTypeAPIKey,
				Platform:    PlatformAntigravity,
				Credentials: map[string]any{"base_url": "https://upstream.example.com"},
			},
			expected: "https://upstream.example.com/antigravity",
		},
		{
			name: "antigravity apikey trims trailing slash",
			account: Account{
				Type:        AccountTypeAPIKey,
				Platform:    PlatformAntigravity,
				Credentials: map[string]any{"base_url": "https://upstream.example.com/"},
			},
			expected: "https://upstream.example.com/antigravity",
		},
		{
			name: "antigravity oauth does NOT append /antigravity",
			account: Account{
				Type:        AccountTypeOAuth,
				Platform:    PlatformAntigravity,
				Credentials: map[string]any{"base_url": "https://upstream.example.com"},
			},
			expected: "https://upstream.example.com",
		},
		{
			name: "oauth without base_url returns default",
			account: Account{
				Type:        AccountTypeOAuth,
				Platform:    PlatformAntigravity,
				Credentials: map[string]any{},
			},
			expected: defaultGeminiURL,
		},
		{
			name: "nil credentials returns default",
			account: Account{
				Type:     AccountTypeAPIKey,
				Platform: PlatformGemini,
			},
			expected: defaultGeminiURL,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.account.GetGeminiBaseURL(defaultGeminiURL)
			if result != tt.expected {
				t.Errorf("GetGeminiBaseURL() = %q, want %q", result, tt.expected)
			}
		})
	}
}
