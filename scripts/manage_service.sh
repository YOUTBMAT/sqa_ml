#!/usr/bin/env bash

# SQA-ML Service Management Script

ACTION=$1
SERVICE_NAME="sqa_ml_bot"
USER_SYSTEMD_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="sqa_ml_bot.service"

case "$ACTION" in
    install)
        echo "Installing $SERVICE_NAME service..."
        mkdir -p "$USER_SYSTEMD_DIR"
        cp "$SERVICE_FILE" "$USER_SYSTEMD_DIR/"
        systemctl --user daemon-reload
        systemctl --user enable "$SERVICE_NAME"
        echo "Service installed and enabled successfully."
        ;;
    start)
        echo "Starting $SERVICE_NAME service..."
        systemctl --user start "$SERVICE_NAME"
        systemctl --user status "$SERVICE_NAME" --no-pager
        ;;
    stop)
        echo "Stopping $SERVICE_NAME service..."
        systemctl --user stop "$SERVICE_NAME"
        ;;
    restart)
        echo "Restarting $SERVICE_NAME service..."
        cp "$SERVICE_FILE" "$USER_SYSTEMD_DIR/"
        systemctl --user daemon-reload
        systemctl --user restart "$SERVICE_NAME"
        systemctl --user status "$SERVICE_NAME" --no-pager
        ;;
    status)
        systemctl --user status "$SERVICE_NAME" --no-pager
        ;;
    logs)
        journalctl --user -u "$SERVICE_NAME" -n 20 --no-pager
        ;;
    *)
        echo "Usage: $0 {install|start|stop|restart|status|logs}"
        exit 1
        ;;
esac
