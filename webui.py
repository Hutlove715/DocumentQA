import gradio as gr
from main import chat_with_agent, load_document

def respond(message, chat_history):
    bot_msg = chat_with_agent(message)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_msg})
    return "", chat_history

def upload_file(file):
    if file is not None:
        load_document(file.name)
        return "✅文档上传并加载完成，可以开始提问！"
    return "⚠️请上传 pdf / txt / docx 文件"

with gr.Blocks(title="RAG文档问答Agent") as demo:
    gr.Markdown("# ■ 文档问答智能助手")
    # 文件上传区域
    with gr.Row():
        file_upload = gr.File(label="上传文档", file_types=[".pdf",".txt",".docx"])
        upload_info = gr.Textbox(label="加载状态", interactive=False)
    file_upload.upload(upload_file, inputs=[file_upload], outputs=[upload_info])

    # 对话区域
    chatbot = gr.Chatbot(height=550)
    msg = gr.Textbox(label="输入问题", placeholder="请输入关于文档的问题...")
    with gr.Row():
        submit_btn = gr.Button("发送")
        clear_btn = gr.Button("清空对话")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
