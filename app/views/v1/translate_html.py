from app.views.v1.translate import *
from bs4 import BeautifulSoup
import random
import os

translate_html = APIRouter(prefix='/api/v1/translate_html')

@translate_html.post('/translate_page', status_code=status.HTTP_200_OK)
async def modify_html_content(request: TranslationRequest):
    
    model_id, src, tgt = fetch_model_data_from_request(request)
    # Parse the HTML content
    soup = BeautifulSoup(request.text, 'html.parser')
    # Modify paragraphs
    for p in soup.find_all('p'):
        p.string = f"{translate_text(model_id, p.get_text(), src, tgt)}"
          
    return TranslationResponse(translation=soup)