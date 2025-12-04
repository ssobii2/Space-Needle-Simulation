# Production Deployment Guide

## Step 1: Set Up Systemd Service (Auto-start)

1. **Copy the service file:**
   ```bash
   sudo cp deploy/needle-app.service /etc/systemd/system/
   ```

2. **Update the paths in the service file if needed:**
   ```bash
   sudo nano /etc/systemd/system/needle-app.service
   ```
   - Verify `WorkingDirectory` matches your app location
   - Verify `ExecStart` path to uvicorn is correct
   - Verify `User` is correct (usually `ubuntu`)

3. **Reload systemd and enable the service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable needle-app
   sudo systemctl start needle-app
   ```

4. **Check status:**
   ```bash
   sudo systemctl status needle-app
   ```

5. **View logs:**
   ```bash
   sudo journalctl -u needle-app -f
   ```

## Step 2: Install and Configure Caddy

Caddy automatically handles SSL certificates, so it's simpler than nginx + certbot.

1. **Install Caddy:**
   ```bash
   sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
   sudo apt update
   sudo apt install -y caddy
   ```

2. **Copy Caddyfile configuration:**
   ```bash
   sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
   ```

3. **Edit the Caddyfile with your domain:**
   ```bash
   sudo nano /etc/caddy/Caddyfile
   ```
   Replace `your-domain.com` with your actual domain name.

4. **Test Caddy configuration:**
   ```bash
   sudo caddy validate --config /etc/caddy/Caddyfile
   ```

5. **Start and enable Caddy:**
   ```bash
   sudo systemctl start caddy
   sudo systemctl enable caddy
   ```

6. **Check Caddy status:**
   ```bash
   sudo systemctl status caddy
   ```

## Step 3: Point Your Domain to EC2

1. **Go to your domain registrar** (GoDaddy, Namecheap, Cloudflare, etc.)

2. **Add DNS records:**
   - Type: `A` | Name: `@` or `your-domain.com` | Value: `your-ec2-public-ip`
   - Type: `A` | Name: `www` | Value: `your-ec2-public-ip`

3. **Wait for DNS propagation** (5-60 minutes, sometimes longer)

4. **Verify DNS is working:**
   ```bash
   dig your-domain.com
   # Should show your EC2 IP
   ```

## Step 4: Update EC2 Security Group

- **Allow inbound:** Port 80 (HTTP) from anywhere (0.0.0.0/0)
- **Allow inbound:** Port 443 (HTTPS) from anywhere (0.0.0.0/0)
- **You can remove port 8000** from public access (Caddy handles it)

## Step 5: Verify Everything Works

1. **Check systemd service:**
   ```bash
   sudo systemctl status needle-app
   ```

2. **Check Caddy:**
   ```bash
   sudo systemctl status caddy
   ```

3. **View Caddy logs:**
   ```bash
   sudo journalctl -u caddy -f
   ```

4. **Test your domain:**
   - Visit `https://your-domain.com` in a browser
   - Caddy will automatically:
     - Get SSL certificate from Let's Encrypt
     - Set up HTTPS
     - Redirect HTTP to HTTPS

**Note:** The first time you visit, Caddy might take 30-60 seconds to get the SSL certificate. Subsequent visits will be instant.

## Troubleshooting

### Service won't start:
```bash
# Check logs
sudo journalctl -u needle-app -n 50

# Check if port 8000 is in use
sudo lsof -i :8000

# Verify paths in service file
cat /etc/systemd/system/needle-app.service
```

### Caddy errors:
```bash
# Check Caddy logs
sudo journalctl -u caddy -n 50

# Validate Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile

# Test Caddy configuration
sudo caddy adapt --config /etc/caddy/Caddyfile
```

### SSL certificate issues:
```bash
# Check Caddy logs for certificate errors
sudo journalctl -u caddy | grep -i cert

# Common issues:
# - DNS not pointing to EC2 IP yet (wait longer)
# - Port 80/443 blocked in security group
# - Domain already has certificate elsewhere (may need to wait)
```

### Can't access from browser:
- Check EC2 security group allows ports 80 and 443
- Verify DNS is pointing to correct IP: `dig your-domain.com`
- Check Caddy is running: `sudo systemctl status caddy`
- Check app is running: `sudo systemctl status needle-app`
- Check Caddy logs: `sudo journalctl -u caddy -f`

### DNS not working:
```bash
# Check if DNS has propagated
dig your-domain.com
nslookup your-domain.com

# If it shows wrong IP or no IP, wait longer or check DNS settings
```

## Maintenance Commands

```bash
# Restart app
sudo systemctl restart needle-app

# Restart Caddy
sudo systemctl restart caddy

# Reload Caddy (after config changes)
sudo systemctl reload caddy

# View app logs
sudo journalctl -u needle-app -f

# View Caddy logs
sudo journalctl -u caddy -f

# Check Caddy configuration
sudo caddy validate --config /etc/caddy/Caddyfile
```

## Caddy Features

- **Automatic HTTPS:** Gets and renews SSL certificates automatically
- **HTTP to HTTPS redirect:** Automatic
- **Certificate renewal:** Happens automatically in background
- **No manual configuration:** Just specify your domain in Caddyfile

## Advanced Caddy Configuration

If you need more control, you can customize the Caddyfile:

```caddy
your-domain.com, www.your-domain.com {
    # Custom headers
    header {
        # Security headers
        -Server
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        X-XSS-Protection "1; mode=block"
    }

    # Reverse proxy
    reverse_proxy 127.0.0.1:8000 {
        header_up Host {host}
        header_up X-Real-IP {remote}
        header_up X-Forwarded-For {remote_host}
        header_up X-Forwarded-Proto {scheme}
    }

    # Compression
    encode gzip zstd

    # Logging
    log {
        output file /var/log/caddy/access.log
    }
}
```

## Comparison: Caddy vs Nginx + Certbot

**Caddy advantages:**
- ✅ Automatic SSL (no certbot needed)
- ✅ Simpler configuration
- ✅ Automatic certificate renewal
- ✅ Single service to manage
- ✅ Built-in HTTP to HTTPS redirect

**Nginx + Certbot advantages:**
- ✅ More control over configuration
- ✅ More widely used (more tutorials)
- ✅ Better for complex setups

For this simple app, Caddy is the better choice!
