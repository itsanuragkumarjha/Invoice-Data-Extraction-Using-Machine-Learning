import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from PIL import Image, ImageDraw
import traceback

import gradio as gr

import torch
from docquery import pipeline
from docquery.document import load_document, ImageDocument
from docquery.ocr_reader import get_ocr_reader



def ensure_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


CHECKPOINTS = {
    "LayoutLMv1": "impira/layoutlm-document-qa",
    "LayoutLMv1 for Invoices": "impira/layoutlm-invoices",
    "Donut": "naver-clova-ix/donut-base-finetuned-docvqa",
}



PIPELINES = {}



def construct_pipeline(task, model):
    global PIPELINES
    if model in PIPELINES:
        return PIPELINES[model]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ret = pipeline(task=task, model=CHECKPOINTS[model], device=device)
    PIPELINES[model] = ret
    return ret


def run_pipeline(model, question, document, top_k):
    pipeline = construct_pipeline("document-question-answering", model)
    return pipeline(question=question, **document.context, top_k=top_k)


# TODO: Move into docquery
# TODO: Support words past the first page (or window?)
def lift_word_boxes(document, page):
    return document.context["image"][page][1]


def expand_bbox(word_boxes):
    if len(word_boxes) == 0:
        return None

    min_x, min_y, max_x, max_y = zip(*[x[1] for x in word_boxes])
    min_x, min_y, max_x, max_y = [min(min_x), min(min_y), max(max_x), max(max_y)]
    return [min_x, min_y, max_x, max_y]


# LayoutLM boxes are normalized to 0, 1000
def normalize_bbox(box, width, height, padding=0.005):
    min_x, min_y, max_x, max_y = [c / 1000 for c in box]
    if padding != 0:
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(max_x + padding, 1)
        max_y = min(max_y + padding, 1)
    return [min_x * width, min_y * height, max_x * width, max_y * height]


examples = [
    [
        "invoice.png",
        "What is the invoice number?",
    ],
    [
        "contract.jpeg",
        "What is the purchase amount?",
    ],
    [
        "statement.png",
        "What are net sales for 2020?",
    ],
    #    [
    #        "docquery.png",
    #        "How many likes does the space have?",
    #    ],
    #    [
    #        "hacker_news.png",
    #        "What is the title of post number 5?",
    #    ],
]

question_files = {
    "What are net sales for 2020?": "statement.pdf",
    "How many likes does the space have?": "https://huggingface.co/spaces/impira/docquery",
    "What is the title of post number 5?": "https://news.ycombinator.com",
}


def process_path(path):
    error = None
    if path:
        try:
            document = load_document(path)
            return (
                document,
                gr.update(visible=True, value=document.preview),
                gr.update(visible=True),
                gr.update(visible=False, value=None),
                gr.update(visible=False, value=None),
                None,
            )
        except Exception as e:
            traceback.print_exc()
            error = str(e)
    return (
        None,
        gr.update(visible=False, value=None),
        gr.update(visible=False),
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=None),
        gr.update(visible=True, value=error) if error is not None else None,
        None,
    )


def process_upload(file):
    if file:
        return process_path(file.name)
    else:
        return (
            None,
            gr.update(visible=False, value=None),
            gr.update(visible=False),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            None,
        )


colors = ["#64A087", "black", "black"]


def process_question(question, document, model=list(CHECKPOINTS.keys())[0]):
    if not question or document is None:
        return None, None, None

    text_value = None
    predictions = run_pipeline(model, question, document, 3)
    pages = [x.copy().convert("RGB") for x in document.preview]
    for i, p in enumerate(ensure_list(predictions)):
        if i == 0:
            text_value = p["answer"]
        else:
            # Keep the code around to produce multiple boxes, but only show the top
            # prediction for now
            break

        if "word_ids" in p:
            image = pages[p["page"]]
            draw = ImageDraw.Draw(image, "RGBA")
            word_boxes = lift_word_boxes(document, p["page"])
            x1, y1, x2, y2 = normalize_bbox(
                expand_bbox([word_boxes[i] for i in p["word_ids"]]),
                image.width,
                image.height,
            )
            draw.rectangle(((x1, y1), (x2, y2)), fill=(0, 255, 0, int(0.4 * 255)))

    return (
        gr.update(visible=True, value=pages),
        gr.update(visible=True, value=predictions),
        gr.update(
            visible=True,
            value=text_value,
        ),
    )


def load_example_document(img, question, model):
    if img is not None:
        if question in question_files:
            document = load_document(question_files[question])
        else:
            document = ImageDocument(Image.fromarray(img), get_ocr_reader())
        preview, answer, answer_text = process_question(question, document, model)
        return document, question, preview, gr.update(visible=True), answer, answer_text
    else:
        return None, None, None, gr.update(visible=False), None, None


CSS = """
#question input {
    font-size: 16px;
}
#url-textbox {
    padding: 0 !important;
}
#short-upload-box .w-full {
    min-height: 10rem !important;
}
/* I think something like this can be used to re-shape
 * the table
 */
/*
.gr-samples-table tr {
    display: inline;
}
.gr-samples-table .p-2 {
    width: 100px;
}
*/
#select-a-file {
    width: 100%;
}
#file-clear {
    padding-top: 2px !important;
    padding-bottom: 2px !important;
    padding-left: 8px !important;
    padding-right: 8px !important;
	margin-top: 10px;
}
.gradio-container .gr-button-primary {
    background: linear-gradient(180deg, #FAED27 0%, #FAED27 100%);
    border: 1px solid #000000;
    border-radius: 8px;
    color: #000000;
}
.gradio-container.dark button#submit-button {
    background: linear-gradient(180deg, #FAED27 0%, #FAED27 100%);
    border: 1px solid #000000;
    border-radius: 8px;
    color: #000000
}
table.gr-samples-table tr td {
    border: none;
    outline: none;
}
table.gr-samples-table tr td:first-of-type {
    width: 0%;
}
div#short-upload-box div.absolute {
    display: none !important;
}
gradio-app > div > div > div > div.w-full > div, .gradio-app > div > div > div > div.w-full > div {
    gap: 0px 2%;
}
gradio-app div div div div.w-full, .gradio-app div div div div.w-full {
    gap: 0px;
}
gradio-app h2, .gradio-app h2 {
    padding-top: 10px;
}
#answer {
    overflow-y: scroll;
    color: white;
    background: #666;
    border-color: #666;
    font-size: 20px;
    font-weight: bold;
}
#answer span {
    color: white;
}
#answer textarea {
    color:white;
    background: #777;
    border-color: #777;
    font-size: 18px;
}
#url-error input {
    color: red;
}
"""

with gr.Blocks(css=CSS) as demo:
    gr.Markdown()
    gr.Markdown(
        
    )

    document = gr.Variable()
    example_question = gr.Textbox(visible=False)
    example_image = gr.Image(visible=False)

    with gr.Row(equal_height=True):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## 1. Select a file", elem_id="select-a-file")
                img_clear_button = gr.Button(
                    "Clear", variant="secondary", elem_id="file-clear", visible=False
                )
            image = gr.Gallery(visible=False)
            with gr.Row(equal_height=True):
                with gr.Column():
                    with gr.Row():
                        url = gr.Textbox(
                            show_label=False,
                            placeholder="URL",
                            lines=1,
                            max_lines=1,
                            elem_id="url-textbox",
                        )
                        submit = gr.Button("Get")
                    url_error = gr.Textbox(
                        visible=False,
                        elem_id="url-error",
                        max_lines=1,
                        interactive=False,
                        label="Error",
                    )
            gr.Markdown("— or —")
            upload = gr.File(label=None, interactive=True, elem_id="short-upload-box")
            gr.Examples(
                examples=examples,
                inputs=[example_image, example_question],
            )

        with gr.Column() as col:
            gr.Markdown("## 2. Ask a question")
            question = gr.Textbox(
                label="Question",
                placeholder="e.g. What is the invoice number?",
                lines=1,
                max_lines=1,
            )
            model = gr.Radio(
                choices=list(CHECKPOINTS.keys()),
                value=list(CHECKPOINTS.keys())[0],
                label="Model",
            )

            with gr.Row():
                clear_button = gr.Button("Clear", variant="secondary")
                submit_button = gr.Button(
                    "Submit", variant="primary", elem_id="submit-button"
                )
            with gr.Column():
                output_text = gr.Textbox(
                    label="Top Answer", visible=False, elem_id="answer"
                )
                output = gr.JSON(label="Output", visible=False)

    for cb in [img_clear_button, clear_button]:
        cb.click(
            lambda _: (
                gr.update(visible=False, value=None),
                None,
                gr.update(visible=False, value=None),
                gr.update(visible=False, value=None),
                gr.update(visible=False),
                None,
                None,
                None,
                gr.update(visible=False, value=None),
                None,
            ),
            inputs=clear_button,
            outputs=[
                image,
                document,
                output,
                output_text,
                img_clear_button,
                example_image,
                upload,
                url,
                url_error,
                question,
            ],
        )

    upload.change(
        fn=process_upload,
        inputs=[upload],
        outputs=[document, image, img_clear_button, output, output_text, url_error],
    )
    submit.click(
        fn=process_path,
        inputs=[url],
        outputs=[document, image, img_clear_button, output, output_text, url_error],
    )

    question.submit(
        fn=process_question,
        inputs=[question, document, model],
        outputs=[image, output, output_text],
    )

    submit_button.click(
        process_question,
        inputs=[question, document, model],
        outputs=[image, output, output_text],
    )

    model.change(
        process_question,
        inputs=[question, document, model],
        outputs=[image, output, output_text],
    )

    example_image.change(
        fn=load_example_document,
        inputs=[example_image, example_question, model],
        outputs=[document, question, image, img_clear_button, output, output_text],
    )

if __name__ == "__main__":
    demo.launch(enable_queue=False)
