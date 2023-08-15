# 导入uvicorn模块。Uvicorn是一个ASGI（Asynchronous Server Gateway Interface）服务器，用于运行基于ASGI标准的Python web应用程序，如FastAPI或Starlette。
import uvicorn

# 从llmtuner模块导入ChatModel类和create_app函数。我们暂时不知道这两者的确切功能，但从名称可以推测，ChatModel可能是与聊天模型相关的类，而create_app可能是用于创建ASGI应用实例的函数。
from llmtuner import ChatModel, create_app

# 定义一个名为main的函数。
def main():
    # 创建一个ChatModel类的实例，并将其存储在chat_model变量中。
    chat_model = ChatModel()
    
    # 调用先前导入的create_app函数，并传入chat_model作为参数。此函数的返回值（可能是一个ASGI应用实例）被存储在app变量中。
    app = create_app(chat_model)
    
    # 使用uvicorn运行上一步创建的ASGI应用。应用将在所有可用的IP地址（0.0.0.0意味着监听所有接口）上的8000端口上运行，并使用1个工作进程。
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
    
    # 当应用开始运行后，在控制台打印一个消息，告诉用户可以访问http://localhost:8000/docs来查看API文档。这通常指的是FastAPI提供的自动生成的Swagger UI文档。
    print("Visit http://localhost:8000/docs for API document.")

# 这是一个Python的常见模式，确保当此脚本作为主程序运行时（而不是作为一个模块导入时）下面的代码会被执行。
if __name__ == "__main__":
    # 调用前面定义的main函数，从而启动整个流程。
    main()
