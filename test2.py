# import os
# # delete jobs on westgrid
# command = 'qdel '
# for i in range(16878874, 16880029):
#     print command + str(i) + '.b0'
# #     os.system(command + str(i) + '.b0')

# file_list = os.listdir('F:/results')
# for file in file_list:
#     print 'java Friedman F:/results/'+file +' > F:/results/' + file[:-4]+'.tex'
    
# import os
# import paramiko
#  
# ssh = paramiko.SSHClient() 
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# # ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
# ssh.connect('bugaboo.westgrid.ca', username='hosseinb', password='hbs25418')
# sftp = ssh.open_sftp()
# sftp.put('C:/Users/hoss/Desktop/test.pdf', 'test.pdf')
# stdin, stdout, stderr = ssh.exec_command('ls')
# for line in stdout:
#     print '... ' + line.strip('\n')
# sftp.close()
# ssh.close()




# import paramiko, base64
# key = paramiko.RSAKey(data=base64.decodestring('AAA...'))
# client = paramiko.SSHClient()
# client.get_host_keys().add('bugaboo.westgrid.ca', 'ssh-rsa', key)
# client.connect('bugaboo.westgrid.ca', username='hosseinb', password='hbs25418')
# stdin, stdout, stderr = client.exec_command('ls')
# for line in stdout:
#     print '... ' + line.strip('\n')
# client.close()


import os
res_path = "~/BCI_Framework/results/BCICIV2a"
res_path = "../results/BCICIV2a"
dir_list = os.listdir(res_path)
dir_list = map((lambda  x: os.path.join(res_path, x)), dir_list)
print dir_list
n_files = len(dir_list)
while n_files > 0: 
    cur_file = list.pop(dir_list)
    if os.path.isdir(cur_file):
        
        dir_list =  map((lambda  x: os.path.join(cur_file, x)), os.listdir(cur_file)) + dir_list
    else:
        
#         new_name = cur_file
#         last_ind = new_name.rfind('_')
#         new_name = new_name[0:last_ind] + '_ALL-1' + new_name[last_ind:] 
#         print cur_file, new_name
#         os.system('mv ' + cur_file + " " + new_name)
#         
        if 'ALL-1' in cur_file:
            print cur_file
            os.system('rm ' + cur_file)
    
    n_files = len(dir_list)
        
