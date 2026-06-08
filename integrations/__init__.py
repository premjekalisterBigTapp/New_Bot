"""
Integrations module for the agentic chatbot.
Provides optional integrations like Zoom live agent support.
"""

try:
    from .zoom.engagement import EngagementManager
    ZOOM_AVAILABLE = True
except ImportError:
    EngagementManager = None
    ZOOM_AVAILABLE = False

__all__ = ["EngagementManager", "ZOOM_AVAILABLE"]
