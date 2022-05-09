# The setup script installs all the required dependencies 
# and create a virtual environment to run the code

import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description='This script is for first time setup on windows machine.\
                                              It creates a python virtual environment and installs \
                                              all the required packages. For help type: python setup.py --help.\
                                              To run this setup script type: python setup.py')
args = parser.parse_args()

def install_cmd(cmd):
    command_run = subprocess.call(cmd, shell=True)
    if command_run == 0:
        # Success
        return 0
    else:
        # Cmd failed
        return 1

if __name__ == "__main__":
    print("Installing python-pip\n")
    pip_cmd = install_cmd("python3 -m ensurepip --upgrade")
    if pip_cmd:
        print("Unable to install python-pip. Terminating ...")
        quit()

    print("Installing python-virtualenv\n")
    pip_cmd = install_cmd("pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org virtualenv")
    if pip_cmd:
        print("Unable to install python-virtualenv. Terminating ...")
        quit()

    print("Creating virtual enviroment virtual-env\n")
    pip_cmd = install_cmd("python3 -m venv virtual-env")
    if pip_cmd:
        print("Unable to craete python virtual environment. Terminating ...")
        quit()

    pip_path = "virtual-env{sep}Scripts{sep}pip3.exe".format(sep=os.sep)
    print("Installing all the required dependencies ...\n")
    pip_cmd = install_cmd("{pip_path} install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt"
                        .format(pip_path=pip_path))
    if pip_cmd:
        print("Unable to install dependencies. Terminating ...")
        quit()

    print("Virtual environment created.\n")
    print("To activate the virtual environment type the following command:\n")
    print("$ virtual-env\\Scripts\\activate")
