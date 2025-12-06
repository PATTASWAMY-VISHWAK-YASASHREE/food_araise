import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.core.vision import analyze_food_image_with_search
from fastapi import HTTPException

@pytest.mark.asyncio
async def test_failsafe_trigger_light():
    """Test that Deep Search=False triggers the LIGHT local model on API failure."""
    with patch("app.core.vision.gemini_client") as mock_gemini, \
         patch("app.core.vision.local_client") as mock_local, \
         patch("app.core.vision._resize_image") as mock_resize:
        
        # Setup: Gemini Fails
        mock_gemini.generate_content.side_effect = Exception("Cloud API Down")
        
        # Setup: Local Client Succeeds
        mock_local.analyze_image.return_value = {"status": "success", "mode": "light"}
        mock_resize.return_value = b"image"
        
        # Test
        result = await analyze_food_image_with_search(b"image", deep_search=False)
        
        # Verify
        assert result == {"status": "success", "mode": "light"}
        mock_local.analyze_image.assert_called_once_with(b"image", deep_search=False)

@pytest.mark.asyncio
async def test_failsafe_trigger_heavy():
    """Test that Deep Search=True triggers the HEAVY local model on API failure."""
    with patch("app.core.vision.gemini_client") as mock_gemini, \
         patch("app.core.vision.local_client") as mock_local, \
         patch("app.core.vision._resize_image") as mock_resize:
        
        # Setup: Gemini Fails
        mock_gemini.generate_content.side_effect = Exception("Cloud API Down")
        
        # Setup: Local Client Succeeds
        mock_local.analyze_image.return_value = {"status": "success", "mode": "heavy"}
        mock_resize.return_value = b"image"
        
        # Test
        result = await analyze_food_image_with_search(b"image", deep_search=True)
        
        # Verify
        assert result == {"status": "success", "mode": "heavy"}
        mock_local.analyze_image.assert_called_once_with(b"image", deep_search=True)

@pytest.mark.asyncio
async def test_all_systems_fail():
    """Test that if both Cloud and Local fail, we get a 500 error."""
    with patch("app.core.vision.gemini_client") as mock_gemini, \
         patch("app.core.vision.local_client") as mock_local, \
         patch("app.core.vision._resize_image"):
        
        mock_gemini.generate_content.side_effect = Exception("Cloud Down")
        mock_local.analyze_image.side_effect = Exception("Local Down")
        
        with pytest.raises(HTTPException) as excinfo:
            await analyze_food_image_with_search(b"image")
        
        assert excinfo.value.status_code == 500
        assert "All AI systems failed" in str(excinfo.value.detail)
