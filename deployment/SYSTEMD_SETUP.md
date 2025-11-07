# PLMS Invariants Checker - systemd Deployment

Production deployment guide for systemd timer.

---

## Quick Setup (Production Server)

```bash
# 1. Copy project to server
rsync -av lnsp-phase-4/ user@server:/opt/lnsp-phase-4/

# 2. Create plms user (optional, for sandboxing)
sudo useradd -r -s /bin/false -d /opt/lnsp-phase-4 plms

# 3. Configure environment
sudo cp /opt/lnsp-phase-4/config/plms_invariants.env /etc/default/plms-invariants
sudo nano /etc/default/plms-invariants
# Set: DB_PATH, SLACK_WEBHOOK_URL, ALERT_EMAIL, PD_ROUTING_KEY, ENV_NAME

# 4. Install systemd units
sudo cp /opt/lnsp-phase-4/deployment/systemd/plms-invariants.service /etc/systemd/system/
sudo cp /opt/lnsp-phase-4/deployment/systemd/plms-invariants.timer /etc/systemd/system/

# 5. Adjust paths in service file if project not at /opt/lnsp-phase-4
sudo nano /etc/systemd/system/plms-invariants.service
# Update WorkingDirectory and ExecStart paths

# 6. Enable and start timer
sudo systemctl daemon-reload
sudo systemd enable --now plms-invariants.timer

# 7. Verify timer is scheduled
systemctl list-timers | grep plms-invariants
# Should show: next run at 02:00 tomorrow

# 8. Test manually (don't wait for 02:00)
sudo systemctl start plms-invariants.service
sudo journalctl -u plms-invariants.service -n 50
```

---

## Verification

```bash
# Check timer status
systemctl status plms-invariants.timer

# Check service logs
journalctl -u plms-invariants.service -f

# Trigger manual run
sudo systemctl start plms-invariants.service

# Check last run result
journalctl -u plms-invariants.service -n 100 --no-pager

# Disable timer (if needed)
sudo systemctl stop plms-invariants.timer
sudo systemctl disable plms-invariants.timer
```

---

## Alternative: Cron (Simpler, No systemd)

If you prefer traditional cron:

```bash
# Edit root crontab
sudo crontab -e

# Add entry (runs at 02:00 local time)
0 2 * * * cd /opt/lnsp-phase-4 && /opt/lnsp-phase-4/scripts/run_invariants.sh >> /var/log/plms/invariants_cron.log 2>&1
```

**Note**: systemd timer is more reliable (persistent, logging, dependencies).

---

## Timezone Configuration

**Important**: Ensure server is in correct timezone for 02:00 ET scheduling.

```bash
# Check current timezone
timedatectl

# Set to Eastern Time (if needed)
sudo timedatectl set-timezone America/New_York

# Verify
date
```

**Alternative**: Use UTC on server, adjust `OnCalendar` in timer:
- 02:00 ET (winter, EST) = 07:00 UTC
- 02:00 ET (summer, EDT) = 06:00 UTC

```ini
# plms-invariants.timer (UTC server)
OnCalendar=*-*-* 07:00:00  # For EST (winter)
```

---

## Alert Configuration

### Slack Webhook

1. Go to: https://api.slack.com/messaging/webhooks
2. Create Incoming Webhook for `#plms-alerts` channel
3. Copy webhook URL
4. Add to `/etc/default/plms-invariants`:
   ```
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
   ```

### Email

Requires sendmail or equivalent MTA:

```bash
# Install sendmail (Debian/Ubuntu)
sudo apt-get install sendmail

# Configure
sudo sendmailconfig

# Test
echo "Test" | sendmail -v your@email.com
```

Add to `/etc/default/plms-invariants`:
```
ALERT_EMAIL=plms-ops@yourdomain.com
```

### PagerDuty

1. Get routing key from PagerDuty integration
2. Add to `/etc/default/plms-invariants`:
   ```
   PD_ROUTING_KEY=your_routing_key_here
   ```
3. Ensure `services/plms/alert_pd.py` exists (created separately)

---

## Sandboxing (Optional, High Security)

Uncomment sandboxing options in `plms-invariants.service`:

```ini
PrivateTmp=yes
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/lnsp-phase-4/logs
ReadWritePaths=/opt/lnsp-phase-4/artifacts
```

Test after enabling:
```bash
sudo systemctl daemon-reload
sudo systemctl start plms-invariants.service
```

---

## Troubleshooting

### Timer not running

```bash
# Check if enabled
systemctl is-enabled plms-invariants.timer

# Check for errors
systemctl status plms-invariants.timer
journalctl -xe -u plms-invariants.timer
```

### Service fails

```bash
# Check service logs
journalctl -u plms-invariants.service -n 100

# Run manually to see errors
cd /opt/lnsp-phase-4
sudo -u plms ./scripts/run_invariants.sh
```

### Alerts not working

```bash
# Check webhook URL is set
cat /etc/default/plms-invariants | grep SLACK_WEBHOOK_URL

# Test Slack manually
curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"Test alert"}' \
    https://hooks.slack.com/services/XXX/YYY/ZZZ
```

### Permission denied

```bash
# Fix ownership
sudo chown -R plms:plms /opt/lnsp-phase-4/logs
sudo chown -R plms:plms /opt/lnsp-phase-4/artifacts

# Fix script permissions
sudo chmod +x /opt/lnsp-phase-4/scripts/run_invariants.sh
```

---

## Monitoring

Add to your monitoring system:

**Check 1**: Timer is active
```bash
systemctl is-active plms-invariants.timer || alert "PLMS timer inactive"
```

**Check 2**: Last run was successful
```bash
systemctl show plms-invariants.service -p ExecMainStatus | grep -q "ExecMainStatus=0" || alert "PLMS check failed"
```

**Check 3**: Ran within last 25 hours
```bash
LAST_RUN=$(systemctl show plms-invariants.service -p ActiveEnterTimestamp | cut -d= -f2)
AGE=$(($(date +%s) - $(date -d "$LAST_RUN" +%s)))
[ $AGE -lt 90000 ] || alert "PLMS check stale"
```

---

**Last Updated**: 2025-11-06
