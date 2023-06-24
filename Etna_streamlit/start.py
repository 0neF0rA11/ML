import subprocess

subprocess.call(["pip", "install", 'pip', 'install', '-r', 'requirements.txt'])
print("Libraries are installed")

# Creating a new process in cmd
process = subprocess.Popen(['cmd'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

# Passing a new command in the process
process.stdin.write('streamlit run main.py\n'.encode())
process.stdin.write('oleyalex-2003@mail.ru\n'.encode())

# Waiting for the process to finish
out, err = process.communicate()

print(out.decode())
