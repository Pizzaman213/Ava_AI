# Fix VS Code Webview Service Worker Error

## This is a BROWSER issue, not a server issue

You're using VS Code Server (remote), so the error is in **your browser**.

## Solutions by Browser:

### Chrome / Edge / Chromium
1. Click the lock icon (or site settings) in the address bar
2. Go to **Cookies and site data**
3. Select **"Allow all cookies"** for this site
4. **OR** enable **"Third-party cookies"** in browser settings for this domain
5. Reload the page (`Ctrl+Shift+R` or `Cmd+Shift+R`)

### Firefox
1. Click the shield icon in the address bar
2. Click **"Turn off Enhanced Tracking Protection for This Site"**
3. Reload the page

### Brave
1. Click the Brave icon (lion) in the address bar
2. Click **"Shields Down"** for this site
3. Reload the page

### Safari
1. Go to Safari > Preferences > Privacy
2. Uncheck **"Prevent cross-site tracking"**
3. Reload the page

## If that doesn't work:

### Clear Browser Cache
1. Press `Ctrl+Shift+Delete` (or `Cmd+Shift+Delete`)
2. Select **"Cached images and files"**
3. Select **"Cookies and other site data"**
4. Click **"Clear data"**
5. Reload VS Code

## Last Resort:
Try a different browser (Chrome recommended for best compatibility)
