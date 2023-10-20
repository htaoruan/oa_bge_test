import mysql.connector
from utils import setup_logger
logger = setup_logger()


class DingoTable:
    def __init__(self,host='172.20.31.10', user='root',passwd='123123', port=3307,database='dingo'):
        self.conn = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=passwd,
        database=database,
        ssl_disabled=True
        )
        self.cursor = self.conn.cursor()
        self.execute("SET WAIT_TIMEOUT=360000000")  ##连接超时时间

    def execute(self, exec_sql):
        self.cursor.execute(exec_sql)
        result = self.cursor.fetchall()
        logger.info(f"table execute {result}")
        # print(result)
        return result
    
    def close(self):
        self.cursor.close()
        self.conn.close()


class BaseTable():
    
    def __init__(self, client, table_name="answer", table_type="answer"):
        self.client = client
        self.table_name = table_name
        self.table_type = table_type
        self.init_sql()
    
    def show_tables(self):
        return self.client.execute("show tables")
        
    def insert(self, data):
        insert_sql = self.insert_sql % data
        self.client.execute(insert_sql)
        max_id = self.client.execute(f"select max(id) from {self.table_name}")
        return max_id[0][0]

    def delete(self, id):
        """根据id删除记录
        """
        delete_sql = self.delete_sql % id
        return self.client.execute(delete_sql)

    def find(self, id):
        find_results = self.client.execute(self.find_sql % id)
        return find_results
    
    def find_all(self):
        """根据关键字段排序并输出全部的内容
        """
        find_results = self.client.execute(self.find_all_sql)
        return find_results
    
    def drop(self):
        return self.client.execute(self.drop_sql)
        

class QaTable(BaseTable):
    def init_sql(self):
        if self.table_type == 'answer':
            self.table_info = ['id', 'answer', 'create_time', 'update_time', 'owner', 'user']
            self.create_sql = f"CREATE TABLE {self.table_name} ({self.table_info[0]} INT AUTO_INCREMENT, {self.table_info[1]} VARCHAR(60000), {self.table_info[2]} VARCHAR(30), {self.table_info[3]} VARCHAR(30), {self.table_info[4]} VARCHAR(20), {self.table_info[5]} VARCHAR(50),PRIMARY KEY({self.table_info[0]}))"
            
            if (self.table_name, ) not in self.show_tables():
                self.client.execute(self.create_sql)
            
            self.insert_sql = f"INSERT INTO {self.table_name}({self.table_info[1]}, {self.table_info[2]}, {self.table_info[3]}, {self.table_info[4]}, {self.table_info[5]}) VALUES ('%s','%s','%s','%s','%s')"
            self.delete_sql = f"DELETE FROM {self.table_name} WHERE id = %d"
            # self.find_sql = f"SELECT * FROM {self.table_name} order by id"
            self.find_sql = f"SELECT * FROM {self.table_name} WHERE id = %d"
            self.find_all_sql = f"SELECT * FROM {self.table_name}"
            
            
        elif self.table_type == 'question':
            self.table_info = ['id', 'question', 'create_time', 'update_time', 'owner', 'answer_id', 'user']
            
            # AUTO_INCREMENT
            self.create_sql = f"CREATE TABLE {self.table_name} ({self.table_info[0]} INT AUTO_INCREMENT, {self.table_info[1]} VARCHAR(50000), {self.table_info[2]} VARCHAR(30), {self.table_info[3]} VARCHAR(30), {self.table_info[4]} VARCHAR(20), {self.table_info[5]} INT, {self.table_info[6]} VARCHAR(20), PRIMARY KEY({self.table_info[0]}))"
            
            if (self.table_name, ) not in self.show_tables():
                self.client.execute(self.create_sql)
            
            self.insert_sql = f"INSERT INTO {self.table_name}({self.table_info[1]},{self.table_info[2]},{self.table_info[3]},{self.table_info[4]},{self.table_info[5]},{self.table_info[6]}) VALUES ('%s','%s','%s','%s','%d','%s')"
            self.delete_sql = f"DELETE FROM {self.table_name} WHERE id = %d"
            # self.find_sql = f"SELECT * FROM {self.table_name} ORDER BY id"
            self.find_sql = f"SELECT * FROM {self.table_name} WHERE id = %d"
            self.find_all_sql = f"SELECT * FROM {self.table_name}"
        
        else:
            print('[ERROR]: No Table %s'%(self.table_type))
        self.drop_sql = f"DROP TABLE {self.table_name}"
        
   


if __name__ == "__main__":
    import json
    
    table_client = DingoTable()
    # qa_client = QaTable(table_client, "question", "question")
    qa_client = QaTable(table_client, table_name="answertest")
    # table_client.execute("SHOW CREATE TABLE answertest")
    # print("insert data:")
    # qa_client.insert(("123", "CREATE&nbsp;TABLE&nbsp;`dim_hbase_class_stu`&nbsp;(<br>&nbsp;&nbsp;`classNo`&nbsp;VARCHAR(2147483647)&nbsp;NOT&nbsp;NULL,<br>&nbsp;&nbsp;`cf`&nbsp;ROW<`className`&nbsp;VARCHAR(2147483647)>,<br>&nbsp;&nbsp;primary&nbsp;key(classNo)&nbsp;NOT&nbsp;ENFORCED<br>)&nbsp;WITH&nbsp;(<br>&nbsp;&nbsp;'connector'&nbsp;=&nbsp;'hbase-2.2',<br>&nbsp;&nbsp;'zookeeper.quorum'&nbsp;=&nbsp;'192.168.7.166:2181',<br>&nbsp;&nbsp;'table-name'&nbsp;=&nbsp;'dim_hbase',<br>&nbsp;&nbsp;'zookeeper.znode.parent'&nbsp;=&nbsp;'/hbase'<br>)".replace("'", "").replace("`", ""), "123", "123", "hch1"))
    # qa_client.insert(("123", "123", "123", "123", "hch2"))
    ##qa_client.insert(("123", "123", "123", json.dumps("哈哈哈"), "hch3"))
    # qa_client.insert(("123", "123", "123", "123", 1, "hch"))
    print("find data:")
    qa_client.find(10)
    print("find all data:")
    res = qa_client.find_all()
    print(31312)
    # print(json.loads(res[4][4]))
    print("delete data")
    qa_client.delete(1)
    print("find all data:")
    qa_client.find_all()
    # print("drop data:")
    # qa_client.drop()
    
    
    
    