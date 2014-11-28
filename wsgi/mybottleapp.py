import os

from bottle import route, default_app
import bottle

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
 
class Person(Base):
    __tablename__ = 'person'
    # Here we define columns for the table person
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    name = Column(String(250), nullable=False)

    def format(self):
        return str(self.id) + " " + self.name

class Address(Base):
    __tablename__ = 'address'
    # Here we define columns for the table address.
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    street_name = Column(String(250))
    street_number = Column(String(250))
    post_code = Column(String(250), nullable=False)
    person_id = Column(Integer, ForeignKey('person.id'))
    person = relationship(Person)
 
# Create an engine that stores data in the local directory's
# sqlalchemy_example.db file.
# engine = create_engine('sqlite:///sqlalchemy_example.db')
# engine = create_engine('mysql://scott:tiger@localhost/foo')
db_user = os.environ['OPENSHIFT_MYSQL_DB_USERNAME']
db_pass = os.environ['OPENSHIFT_MYSQL_DB_PASSWORD']
db_credentials = db_user + ":" + db_pass
db_host = os.environ['OPENSHIFT_MYSQL_DB_HOST']
db_port = os.environ['OPENSHIFT_MYSQL_DB_PORT']
db_name = "pepapot"
db_url = db_host + ":" + db_port + "/" + db_name
db_protocol = 'mysql+mysqlconnector://'
db_engine_string = db_protocol + db_credentials + "@" + db_url
engine = create_engine(db_engine_string)

Base.metadata.bind = engine

@route('/name/<name>')
def nameindex(name='Stranger'):
    return '<strong>Hello, %s!</strong>' % name
 
@route('/')
def index():
    DBSession = sessionmaker()
    DBSession.bind = engine
    session = DBSession()
    # Make a query to find all Persons in the database
    persons = session.query(Person).all()
    persons_string = ", ".join([p.format() for p in persons])

    return '<strong>Hello World!</strong>' + persons_string

@route('/add/<name>')
def add_person(name='Stranger'):
    DBSession = sessionmaker()
    DBSession.bind = engine
    session = DBSession()
    new_person = Person(name=name)
    session.add(new_person)
    session.commit()
    return """To see the new database <a href="/">click here</a>"""

application=default_app()

if __name__ == '__main__':
    import sys
    if "init-db" in sys.argv:
        Base.metadata.create_all(engine)
    else:
        bottle.debug()
        bottle.run()
