import base64
import io
import json
import os

import dotenv
import gradio as gr
import requests
import retrying
from PIL import Image


@retrying.retry(stop_max_attempt_number=2, wait_fixed=1000)
def generate_image(image, inference_steps, guidance_scale, image_size, prompt, model_choice, style_name):
    # 检查模型名称是否包含SDXL-Lightning
    if "SDXL-Lightning" in model_choice:
        inference_steps = 4
        guidance_scale = 1

    if "turbo" in model_choice:
        inference_steps = 6
        guidance_scale = 1

    image_size = {
        '1:1': '1024x1024',
        '1:2': '1024x2048',
        '3:2': '1536x1024',
        '3:4': '1536x2048',
        '16:9': '2048x1152',
        '9:16': '1152x2048',
    }[image_size]

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        'Authorization': 'Bearer ' + os.getenv('SILICONFLOW_API_KEY'),
    }

    data = {
        "prompt": prompt,
        "image_size": image_size,
        "batch_size": 1,
        "num_inference_steps": inference_steps,
        "guidance_scale": guidance_scale
    }

    # 如果是图生图模型，并且提供了图像，则添加图像数据
    if "image-to-image" in model_choice and image is not None:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        data["image"] = base64_string
    elif "image-to-image" in model_choice and image is None:
        # 如果选择了图生图模型但没有提供图像，可以返回一个错误消息或使用默认图像
        return Image.new('RGB', (512, 512), color='white')  # 创建一个白色的默认图像

    if "PhotoMaker" in model_choice:
        data["style_name"] = style_name
        data["style_strengh_radio"] = 20

    url = f"https://api.siliconflow.cn/v1/{model_choice}"

    with requests.post(url, data=json.dumps(data), headers=headers) as response:
        res = response.json()
        image_url = res['images'][0]['url']
        with requests.get(image_url) as image_response:
            img = Image.open(io.BytesIO(image_response.content))
            return img


def dosomething(image, inference_steps, guidance_scale, image_size, prompt, model_choice, style_name):
    return generate_image(image, inference_steps, guidance_scale, image_size, prompt, model_choice, style_name)


if __name__ == "__main__":
    dotenv.load_dotenv()

    with gr.Blocks() as demo:
        gr.Markdown("# 文生图、图生图")

        guidance_scale = gr.Slider(minimum=0, maximum=100, value=7.5, step=0.1, label='引导比例')
        with gr.Row():
            inference_steps = gr.Number(minimum=1, maximum=100, value=20, label='推理步数')
            image_size = gr.Dropdown(['1:1', '1:2', '3:2', '3:4', '16:9', '9:16'], label='生成图片比例', value='1:1')
            style_name = gr.Dropdown([
                'Cinematic',
                'Comic book',
                'Disney Character',
                'Digital Art',
                'Photographic (Default)',
                'Fantasy Art',
                'Neopunk',
                'Enhance',
                'Lowpoly',
                'Line art',
                '(No style)'
            ], label='PhotoMaker风格', value='Photographic (Default)')

            model_choice = gr.Dropdown([
                'black-forest-labs/FLUX.1-schnell/text-to-image',
                'stabilityai/stable-diffusion-3-medium/text-to-image',
                'stabilityai/stable-diffusion-xl-base-1.0/text-to-image',
                'stabilityai/stable-diffusion-2-1/text-to-image',
                'stabilityai/sd-turbo/text-to-image',
                'stabilityai/sdxl-turbo/text-to-image',
                'ByteDance/SDXL-Lightning/text-to-image',
                'stabilityai/stable-diffusion-xl-base-1.0/image-to-image',
                'stabilityai/stable-diffusion-2-1/image-to-image',
                'ByteDance/SDXL-Lightning/image-to-image',
                'TencentARC/PhotoMaker/image-to-image',
            ], label='选择模型', value='stabilityai/stable-diffusion-xl-base-1.0/image-to-image')

        prompt_input = gr.Textbox(label="输入prompt", lines=3,
                                  value="Transform all objects in the scene into a highly detailed and realistic anime style. Ensure that all characters have perfectly proportioned features including complete and natural-looking hands and fingers, and symmetrical, well-defined facial features with no distortions or anomalies. All objects should be rendered with vibrant and colorful details, smooth shading, and dynamic compositions. The style should resemble the works of Studio Ghibli or Makoto Shinkai, with meticulous attention to detail in every aspect, including backgrounds, clothing, and accessories. The overall image should be cohesive, with a harmonious blend of all elements.",
                                  placeholder="在此输入你的prompt...")

        with gr.Row():
            # with gr.Column():
            clear_btn = gr.Button("清空")
            generate_btn = gr.Button("生成图片")

        with gr.Row():
            image_input = gr.Image(label='选择原图 (仅用于图生图)', type='pil')
            output_image = gr.Image(label='生成的图片')

        generate_btn.click(
            fn=dosomething,
            inputs=[image_input, inference_steps, guidance_scale, image_size, prompt_input, model_choice, style_name],
            outputs=output_image
        )

        clear_btn.click(
            fn=lambda: (None, 20, 5, "1:1",
                        "a half-body portrait of a man img wearing the sunglasses in Iron man suit, best quality",
                        "TencentARC/PhotoMaker/image-to-image", "Photographic (Default)", None),
            inputs=None,
            outputs=[image_input, inference_steps, guidance_scale, image_size, prompt_input, model_choice, style_name,
                     output_image]
        )

    demo.launch(server_name='0.0.0.0', server_port=7861)