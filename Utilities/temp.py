
# def get_vals(): return ['abcd',10,20,30,40,'a']
f = open('dump_file.xml','a')
while(1):
    file_name,top,left,width,height,label = get_vals()
    f.write("<image file='%s'>\n\t<box top='%d' left='%d' width='%d' height='%d'>\n\t\t<label>%s</label>\n\t</box>\n</image>\n"%(file_name,top,left,width,height,label))
