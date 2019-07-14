from MysqlDB import MyDB
import os
import datetime
uptime = 0
with MyDB('localhost','root','123456','student') as cs:
    
    sql = "select * from shijian where id='1'"
    try:
        cs.execute(sql)
        result = cs.fetchall()
        for row in result:
            uptime = row[1]
    except:
        cs.rollback()

print(uptime)
