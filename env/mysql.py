import numpy as np
import pymysql
ti_obj = pymysql.connect(
    host='172.23.46.214',             # MySQL服务端的ip地址
    port=3306,                    # MySQL默认的port地址(端口号)
    user='root',                  # 用户名
    passwd='test123',               # 密码
    database='mysql',               # 库名
    charset='utf8'                # 字符编码 类型
)
cursor = ti_obj.cursor()
Q=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
def firstSet():
    sql1 = 'UPDATE me1_2 SET number=1, time=1'
    ti_obj.ping(reconnect=True)
    cursor.execute(sql1)
    ti_obj.commit()
    sql2 = 'update me2_2 SET emptytime=-1, res1=0.1'
    ti_obj.ping(reconnect=True)
    cursor.execute(sql2)
    ti_obj.commit()
def pagerank(G,alpha,max_iter,tol):
    n= G.shape[0]
    v= np.ones(n)/n
    last_v = np.ones(n)*np.inf
    G_T = G.transpose()
    sql="SELECT emptytime from me2_2 order by meid"
    ti_obj.ping(reconnect=True)
    cursor.execute(sql)
    infolist=cursor.fetchall()
    ti_obj.commit()
    VS=[]
    sum=0.1
    for i in range(len(infolist)):
        if infolist[i][0] == -1:
            VS.append(0)
        else:
            VS.append(infolist[i][0])
            sum+=infolist[i][0]
    VS=np.array(VS)
    for i in range(max_iter):
        v=(1-alpha)*VS/sum+alpha*G_T.dot(v)
        if np.abs(v-last_v).sum()<tol:
            break
        last_v = v
    for i in range(len(v)):
        sql = "update me2_2 SET res1=%lf WHERE meid='%s'" % (v[i], Q[i])
        ti_obj.ping(reconnect=True)
        cursor.execute(sql)
        ti_obj.commit()
    return v
def run():
    for i in range(len(Q)):
        sql="SELECT CAST(SUM(number) AS SIGNED), CAST(SUM(time) AS SIGNED)FROM me1_2 WHERE tos='%s'" % Q[i]
        cursor.execute(sql)
        sum1 = cursor.fetchall()
        sql="update me1_2 a,me1_2 b SET a.RPR=0.5*b.number/%d+0.5*b.time/%d WHERE a.froms=b.froms and a.tos=b.tos AND a.tos='%s'" % (sum1[0][0], sum1[0][1], Q[i])
        cursor.execute(sql)
        ti_obj.commit()
    list=[]
    for i in range(len(Q)):
        sql3 = "SELECT RPR FROM me1_2 WHERE tos='%s' order by froms" % Q[i]
        ti_obj.ping(reconnect=True)
        cursor.execute(sql3)
        result = cursor.fetchall()
        ti_obj.commit()
        L=[]
        for j in range(len(result)):
            L.append(result[j][0])
        list.append(L)
    G=np.array(list)
    pagerank(G,0.5,100,0.00001)
    ti_obj.close()

def Createtable():
    for i in range(len(Q)):
        for j in range(len(Q)):
            sql = "INSERT INTO me1_2(froms,tos) VALUES('%s','%s')" % (Q[i], Q[j])
            ti_obj.ping(reconnect=True)
            cursor.execute(sql)
            ti_obj.commit()