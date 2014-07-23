from bottle import route, default_app
import bottle

@route('/name/<name>')
def nameindex(name='Stranger'):
    return '<strong>Hello, %s!</strong>' % name
 
@route('/')
def index():
    return '<strong>Hello World!</strong>'

application=default_app()

if __name__ == '__main__':
    bottle.debug()
    bottle.run()
