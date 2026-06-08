#!/usr/bin/env bash
# restart_bot.sh — reliably restart the bot from the correct folder.
#
# Usage (run from ~/new_bot/New_Bot on the server):
#   bash restart_bot.sh
#
# What it does:
#   1. Kills the existing bot_app tmux session (if running)
#   2. Creates a fresh bot_app session running python3 main.py from this folder
#   3. Optionally restarts the ngrok tunnel (uncomment the ngrok block)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="bot_app"
NGROK_SESSION="bot_ngrok"

echo "=== BigTapp Bot Restart ==="
echo "Bot folder: $SCRIPT_DIR"

# Kill existing bot session if it exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Stopping existing tmux session: $SESSION"
    tmux kill-session -t "$SESSION"
fi

# Start a fresh bot session
echo "Starting bot in tmux session: $SESSION"
tmux new -d -s "$SESSION" "cd '$SCRIPT_DIR' && python3 main.py"

echo ""
echo "Bot started. Attach with:  tmux attach -t $SESSION"
echo "Detach with:               Ctrl+B, then D"
echo ""

# Optionally restart ngrok (uncomment if needed)
# if tmux has-session -t "$NGROK_SESSION" 2>/dev/null; then
#     echo "Restarting ngrok session: $NGROK_SESSION"
#     tmux kill-session -t "$NGROK_SESSION"
# fi
# tmux new -d -s "$NGROK_SESSION" "cd '$SCRIPT_DIR' && ./ngrok http 8000"
# echo "ngrok started. Attach with:  tmux attach -t $NGROK_SESSION"

echo "Done."
