INSTRUCTION HOW TO BUILD THE STANDALONE APP

1. Create '*.py' file which you want to pack into the executable installer.

2. Install the Pyinstaller library (use the "pip install pyinstaller" command).

3. If your file can be run without using the command line, then write the command 
"pyinstaller /path/to/yourscript.py".

4. If you need a console to run your code, create a new file in which you pass the command to run 
and repeat the 3rd step.
	Code example:
		import subprocess

		# Creating a new process in cmd
		process = subprocess.Popen(['cmd'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

		# Passing a new command in the process
		process.stdin.write('streamlit run yourscript.py\n'.encode())

		# Waiting for the process to finish
		out, err = process.communicate()

		print(out.decode())
	
	IMPORTANT NOTE:
	If you are creating a start file, your '.exe' file and the script must be in the same directory.