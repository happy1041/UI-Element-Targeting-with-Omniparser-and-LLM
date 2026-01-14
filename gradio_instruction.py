import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
import re
import json
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
from openai import OpenAI

# 1. Initialization
print("Initializing Models...")

# Clear proxy environment variables to prevent API connection issues
for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    os.environ.pop(var, None)

yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")

def encode_image(image):
    buffered = io.BytesIO()
    # High quality for visual grounding
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_combined_decision(api_key, base_url, model_name, elements_text, instruction, image_pil):
    # Ensure base_url ends with /v1
    if base_url and "api" in base_url and not base_url.endswith("/v1") and not base_url.endswith("/v1/"):
        base_url = base_url.rstrip("/") + "/v1"
    
    print(f"Connecting to Multimodal LLM at: {base_url} using model: {model_name}")
    client = OpenAI(api_key=api_key, base_url=base_url)
    base64_image = encode_image(image_pil)
    
    # Advanced ID selection prompt with OCR Correction Logic
    system_prompt = f"""You are a UI Visual Agent. Your objective is to pick the single most relevant element ID from a provided list based on a natural language instruction and a screenshot.

### IMPORTANT: OCR CORRECTION ###
OCR often misinterprets icons as simple text. 
- If an element is type 'icon' but the content is '三', it is likely a HAMBURGER MENU.
- If content is '口', it might be a SQUARE BUTTON or CHECKBOX.
- If content is 'X', it is likely a CLOSE button.
ALWAYS look at the SCREENSHOT to verify what the icon actually looks like.

### TASK ###
1. Map the user's instruction to the VISUAL appearance in the screenshot.
2. Find the ID in the 'OMNI-DETECTION LIST' that covers that visual area.
3. Trust your VISUAL perception over the OCR text if they conflict.

### OUTPUT FORMAT (STRICT JSON) ###
You MUST return a JSON object with EXACTLY these two keys:
{{
  "reasoning": "Briefly explain the target visual match",
  "id": 123
}}
Do NOT use other keys like 'target_id' or 'action'.

### TASK ###
1. Map the user's instruction to the VISUAL appearance in the screenshot.
2. Find the ID in the 'OMNI-DETECTION LIST' that covers that visual area.
3. Trust your VISUAL perception over the OCR text if they conflict.
"""

    prompt = f"Target Instruction: '{instruction}'\n\nOMNI-DETECTION LIST:\n{elements_text}\n\nDecision: Return only the JSON object with keys 'reasoning' and 'id'."
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        print(f"LLM Response:\n{content}")
        
        # Parse JSON
        json_match = re.search(r"(\{.*\})", content, re.DOTALL)
        if json_match:
            try:
                res_dict = json.loads(json_match.group(1))
                # Robust parsing for field name variations
                if "reasoning" not in res_dict:
                    res_dict["reasoning"] = res_dict.get("reason", res_dict.get("reasoning", res_dict.get("action", "No reasoning provided.")))
                
                # Try multiple common ID field names
                possible_id_keys = ["id", "target_id", "element_id", "matched_id"]
                found_id = None
                for k in possible_id_keys:
                    if k in res_dict and isinstance(res_dict[k], (int, float)):
                        found_id = int(res_dict[k])
                        break
                
                if found_id is not None:
                    res_dict["id"] = found_id
                else:
                    # Generic regex fallback to find any field ending in "id" or just "id"
                    # Matches "id": 52 or "target_id": 52
                    id_search = re.search(r"\"[\w_]*id\"\s*:\s*(\d+)", content, re.IGNORECASE)
                    if id_search:
                        res_dict["id"] = int(id_search.group(1))
                
                return res_dict
            except json.JSONDecodeError:
                pass
        
        # Fallback heuristic for ID if JSON fails completely
        id_match = re.search(r"\"?[\w_]*id\"?\s*:\s*(\d+)", content, re.IGNORECASE)
        if id_match:
            return {"reasoning": "Heuristic match from raw text", "id": int(id_match.group(1))}
            
        return {"reasoning": f"Failed to extract ID from: {content[:100]}", "id": -1}
    except Exception as e:
        print(f"LLM Error: {e}")
        return {"reasoning": f"API Error: {e}", "id": -1}

def process_instruction(
    image_input,
    instruction,
    api_key,
    base_url,
    model_name,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz
):
    # --- Step 1: OmniParser Core ---
    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    # Important: Use False for PaddleOCR in this specific environment
    ocr_bbox_rslt, _ = check_ocr_box(image_input, display_img=False, output_bb_format='xyxy', easyocr_args={'paragraph': False, 'text_threshold':0.5}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_input, yolo_model, BOX_TRESHOLD=box_threshold, output_coord_in_ratio=True, 
        ocr_bbox=ocr_bbox, draw_bbox_config=draw_bbox_config, 
        caption_model_processor=caption_model_processor, ocr_text=text,
        iou_threshold=iou_threshold, imgsz=imgsz
    )

    # Convert OmniParser's base64 output back to PIL Image for display
    all_boxes_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))

    # --- Step 2: Format for LLM ---
    elements_text = ""
    for i, item in enumerate(parsed_content_list):
        content = item.get('content', 'No description').strip()
        etype = item.get('type', 'unknown')
        bbox = item.get('bbox', [0,0,0,0]) # [x1, y1, x2, y2] ratio
        
        # UI Pattern: If it's an icon and OCR is a single character, it's a high-risk mis-OCR
        is_risky = (etype == "icon" and len(content) == 1)
        risk_tag = "[Low Confidence OCR]" if is_risky else ""
        
        bbox_str = f"[{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]"
        interact = "Interactive" if item.get('interactivity', True) else "Static"
        
        if content:
            # Highlight risky OCR to the LLM
            elements_text += f"ID {i} {bbox_str} [{etype} | {interact}]: {content} {risk_tag}\n"

    # --- Step 3: Call LLM ---
    print("Requesting decision from LLM (ID Focused)...")
    decision = get_combined_decision(api_key, base_url, model_name, elements_text, instruction, image_input)
    
    reason = decision.get("reasoning", "No reasoning provided.")
    best_id = decision.get("id", -1)

    # --- Step 4: Visualize ---
    result_image = image_input.copy()
    draw = ImageDraw.Draw(result_image)
    w, h = result_image.size
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()

    # helper to draw ID box (Rectangle only)
    def draw_id_box(target_id, img_draw, w_img, h_img, d_font):
        if 0 <= target_id < len(parsed_content_list):
            bbox = parsed_content_list[target_id]['bbox']
            pixel_bbox = [bbox[0]*w_img, bbox[1]*h_img, bbox[2]*w_img, bbox[3]*h_img]
            img_draw.rectangle(pixel_bbox, outline="red", width=8)
            # Text label removed per user request
            return True
        return False

    selected_bbox_val = "{}"
    if draw_id_box(best_id, draw, w, h, font):
        # Retrieve the chosen element's bbox
        target_item = parsed_content_list[best_id]
        t_bbox = target_item.get('bbox', [0,0,0,0])
        bbox_data = {
            "id": best_id,
            "bbox_normalized": [round(float(x), 3) for x in t_bbox],
            "type": target_item.get('type', 'unknown'),
            "description": target_item.get('content', '')
        }
        selected_bbox_val = json.dumps(bbox_data, indent=2, ensure_ascii=False)
        status_msg = f"Final Decision: ID {best_id} | Reason: {reason}"
    else:
        status_msg = f"No ID Selected or Invalid ID ({best_id}). Reason: {reason}"

    return result_image, all_boxes_image, elements_text, status_msg, selected_bbox_val

# --- UI Layout ---
with gr.Blocks(title="OmniParser Instruction Follower") as demo:
    gr.Markdown("# OmniParser + LLM API: UI Instruction Follower")
    
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type='pil', label='Step 1: Upload Screenshot')
            instr_in = gr.Textbox(label='Step 2: Natural Language Instruction', placeholder='e.g. Click the login button')
            
            with gr.Accordion("API Settings", open=True):
                api_key_in = gr.Textbox(label='API Key', type='password', value='sk-...')
                base_url_in = gr.Textbox(label='Base URL', value='https:')
                model_in = gr.Textbox(label='Model Name', value='gemini-3-flash-preview-128')
                
            with gr.Accordion("Advanced OmniParser Config", open=False):
                box_hit = gr.Slider(0.005, 1.0, 0.03, label='Box Threshold (越小图标越多)')
                iou_hit = gr.Slider(0.01, 1.0, 0.15, label='IOU Threshold (重叠度阈值)')
                ocr_check = gr.Checkbox(label='Use PaddleOCR (Broken in Env)', value=True)
                size_hit = gr.Slider(640, 1920, 1280, step=32, label='Detect Image Size (越高越能识别小图标)')
            
            btn = gr.Button("Find Element", variant='primary')
            
        with gr.Column():
            image_out = gr.Image(type='pil', label='Step 3: Target Found (Red Box = Identified)')
            status_out = gr.Textbox(label='Status', lines=3)
            with gr.Group():
                gr.Markdown("### Selected Element BBox")
                bbox_res = gr.Code(label=None, language="json", interactive=False)
            all_boxes_out = gr.Image(type='pil', label='OmniParser Analysis View (All Detects)')
            elements_out = gr.Textbox(label='Raw Parsed Elements', lines=10)

    btn.click(
        fn=process_instruction,
        inputs=[image_in, instr_in, api_key_in, base_url_in, model_in, box_hit, iou_hit, ocr_check, size_hit],
        outputs=[image_out, all_boxes_out, elements_out, status_out, bbox_res]
    )

demo.launch(share=True, server_port=7862, server_name='0.0.0.0')