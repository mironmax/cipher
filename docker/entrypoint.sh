#!/bin/sh
set -e

# Start background log rotation daemon
# Runs logrotate daily at 3 AM without needing cron permissions
(
    while true; do
        # Calculate seconds until 3 AM
        current_hour=$(date +%H)
        current_min=$(date +%M)
        current_sec=$(date +%S)

        # If it's 3 AM (within a 5-minute window), run logrotate
        if [ "$current_hour" -eq 3 ] && [ "$current_min" -lt 5 ]; then
            /usr/sbin/logrotate -s /app/data/logs/logrotate.status /etc/logrotate.d/cipher >> /app/data/logs/logrotate.log 2>&1 || true
            # Sleep for 10 minutes to avoid running multiple times
            sleep 600
        fi

        # Check every 5 minutes
        sleep 300
    done
) &

echo "Logrotate daemon started (runs daily at 3 AM)"

# Execute the original command (passed as arguments)
exec "$@"