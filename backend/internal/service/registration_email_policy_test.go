//go:build unit

package service

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestValidateRegistrationGmailEmail(t *testing.T) {
	require.NoError(t, ValidateRegistrationGmailEmail("user_name@example.com"))
	require.NoError(t, ValidateRegistrationGmailEmail("not-an-email"))
	require.NoError(t, ValidateRegistrationGmailEmail("user@localhost"))
	require.NoError(t, ValidateRegistrationGmailEmail("User_123@gmail.com"))
	require.NoError(t, ValidateRegistrationGmailEmail("user_name@gmail.com"))

	require.ErrorIs(t, ValidateRegistrationGmailEmail("User.Name@gmail.com"), ErrInvalidGmailAddress)
	require.ErrorIs(t, ValidateRegistrationGmailEmail("user+tag@gmail.com"), ErrInvalidGmailAddress)
}

func TestNormalizeRegistrationEmailSuffixWhitelist(t *testing.T) {
	got, err := NormalizeRegistrationEmailSuffixWhitelist([]string{"example.com", "@EXAMPLE.COM", " @foo.bar "})
	require.NoError(t, err)
	require.Equal(t, []string{"@example.com", "@foo.bar"}, got)
}

func TestNormalizeRegistrationEmailSuffixWhitelist_Invalid(t *testing.T) {
	_, err := NormalizeRegistrationEmailSuffixWhitelist([]string{"@invalid_domain"})
	require.Error(t, err)
}

func TestParseRegistrationEmailSuffixWhitelist(t *testing.T) {
	got := ParseRegistrationEmailSuffixWhitelist(`["example.com","@foo.bar","@invalid_domain"]`)
	require.Equal(t, []string{"@example.com", "@foo.bar"}, got)
}

func TestIsRegistrationEmailSuffixAllowed(t *testing.T) {
	require.True(t, IsRegistrationEmailSuffixAllowed("user@example.com", []string{"@example.com"}))
	require.False(t, IsRegistrationEmailSuffixAllowed("user@sub.example.com", []string{"@example.com"}))
	require.True(t, IsRegistrationEmailSuffixAllowed("user@any.com", []string{}))
}
