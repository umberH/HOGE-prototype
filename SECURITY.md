# Security & Privacy Guide

## User API Key Security

### How It Works

When users provide their OpenAI API key in the Streamlit dashboard:

1. **Input is masked** - `type="password"` ensures the key is not visible on screen
2. **Session-only storage** - Key is stored in `st.session_state`:
   - ✅ Only exists during the user's browser session
   - ✅ **Never** saved to disk
   - ✅ **Never** logged or stored in database
   - ✅ **Not** shared between users
   - ✅ Automatically deleted when browser is closed
3. **HTTPS encryption** - All data transmitted over encrypted connection (Streamlit Cloud uses HTTPS)
4. **No server-side storage** - The key is never stored on the server

### Why This Is Safe

**Streamlit Session State is Client-Side:**
- Each user's session is isolated
- Session data lives in the browser's memory
- No persistence between sessions
- No cross-user contamination

**HTTPS Encryption:**
- All communication between browser and server is encrypted
- API key is encrypted in transit to OpenAI
- No man-in-the-middle attacks possible

**No Logging:**
- The application does not log API keys
- OpenAI API calls use the user's key directly
- No intermediate storage or caching of keys

### Best Practices for Users

1. **Create a dedicated API key:**
   - Go to https://platform.openai.com/api-keys
   - Create a new key specifically for this app
   - Set a recognizable name (e.g., "HOGE Dashboard")

2. **Set usage limits:**
   - In your OpenAI account, set monthly usage limits
   - This prevents unexpected charges
   - Recommended: Start with $5-10/month limit

3. **Monitor usage:**
   - Check your OpenAI usage dashboard regularly
   - Review API calls and costs
   - Set up usage alerts in OpenAI account

4. **Revoke when done:**
   - If you no longer use the app, revoke the API key
   - This ensures no future unauthorized use
   - You can always create a new key later

5. **Don't share screenshots:**
   - If taking screenshots, ensure API key is not visible
   - The input is masked, but be cautious

### For Developers

**Additional Security Measures Implemented:**

1. **No environment variable fallback in production:**
   - The app requires user-provided API keys
   - No default/shared API key in deployment
   - Each user brings their own credentials

2. **Input validation:**
   - API key format is validated before use
   - Invalid keys result in clear error messages
   - No silent failures or key leakage in logs

3. **Error handling:**
   - OpenAI API errors are caught and sanitized
   - Error messages don't expose API keys
   - Users see helpful, non-sensitive error information

4. **Session management:**
   - API key is cleared when user reloads page (unless re-entered)
   - No persistent cookies or local storage used
   - Clean session on browser close

### Comparison with Alternatives

| Method | Security | Pros | Cons |
|--------|----------|------|------|
| **User-provided key (our method)** | ⭐⭐⭐⭐⭐ | User controls their key, no shared secrets, isolated usage | User must have OpenAI account |
| **Shared API key** | ⭐⭐ | Easy for users | Risky, expensive, hard to track usage |
| **OAuth flow** | ⭐⭐⭐⭐⭐ | Official auth | Complex to implement, requires OpenAI OAuth (not available) |
| **API key stored in database** | ⭐⭐⭐ | Convenient | Requires encryption, trust in app security |

### What We DON'T Do

❌ **We do NOT:**
- Store API keys in database
- Log API keys to files
- Share keys between users
- Send keys to third parties (except OpenAI API)
- Cache or persist keys
- Expose keys in error messages
- Include keys in URLs or query parameters

✅ **We DO:**
- Use HTTPS for all communication
- Store keys only in session memory
- Clear keys on session end
- Validate API keys before use
- Provide clear security information to users
- Follow OpenAI's security best practices

### Compliance

This approach aligns with:
- **GDPR** - No personal data stored, user has full control
- **OWASP Top 10** - Secure credential management
- **OpenAI Terms of Service** - Proper API key handling
- **Industry best practices** - Minimal data retention

### Incident Response

If you suspect your API key has been compromised:

1. **Immediately revoke the key** at https://platform.openai.com/api-keys
2. **Review usage** in OpenAI dashboard for unauthorized calls
3. **Create a new key** if you want to continue using the app
4. **Enable usage limits** to prevent abuse

### Questions?

- **Q: Can the app developers see my API key?**
  - A: No. The key is only in your browser's memory, never sent to our servers except encrypted to OpenAI.

- **Q: What happens if I refresh the page?**
  - A: You'll need to re-enter your API key. It's not saved.

- **Q: Can other users access my API key?**
  - A: No. Each session is completely isolated.

- **Q: Is my usage tracked?**
  - A: Only by OpenAI in your account. We don't track or log API calls.

- **Q: What if someone hacks the Streamlit server?**
  - A: Your API key is not stored on the server, only in your browser session. An attacker would need to hijack your active session (protected by HTTPS).

### Report Security Issues

If you discover a security vulnerability, please report it responsibly:

- **Email**: [your-security-email@example.com]
- **GitHub**: Create a private security advisory
- **Do NOT** open a public issue for security vulnerabilities

---

**Summary**: User-provided API keys in session state is a secure, privacy-respecting approach that gives users full control over their credentials and usage.
