#!/bin/bash
# Health check script for all GPU services

HEALTHY=true

for PORT in 8001 8002 8003 8004; do
    if ! curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        HEALTHY=false
        echo "UNHEALTHY: port $PORT"
    fi
done

if [ "$HEALTHY" = true ]; then
    echo "ALL HEALTHY"
    exit 0
else
    exit 1
fi
