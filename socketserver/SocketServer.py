import tornado.websocket
import tornado.web
import tornado.ioloop
 
 
class EchoWebSocket(tornado.websocket.WebSocketHandler):
    clients = []
    def check_origin(self, origin):
        return True
 
    def data_received(self, chunk):
        pass
 
    def open(self):
        EchoWebSocket.clients.append(self)
        self.write_message("WebSocket opened")
 
    def on_message(self, message):
        for client in EchoWebSocket.clients:
            client.write_message(u"You said: " + message)
 
    def on_close(self):
        EchoWebSocket.clients.remove(self)
        print("WebSocket closed")
 
if __name__=='__main__':
    app = tornado.web.Application([
        (r"/", EchoWebSocket)
    ])
    app.listen(8881)
    tornado.ioloop.IOLoop.instance().start()