import sys
import pip
import importlib


def install_package_if_required(import_name: str, package_name: str):
    try:
        importlib.import_module(import_name)
    
    except:
        try:
            # https://blender.stackexchange.com/a/153520
            import sys
            import subprocess

            py_exec     = sys.executable
            py_prefix   = sys.exec_prefix

            # ensure pip is installed & update
            subprocess.call([str(py_exec), "-m", "ensurepip", "--user"])
            subprocess.call([str(py_exec), "-m", "pip", "install", "--target={}".format(py_prefix), "--upgrade", "pip"])

            # install dependencies using pip
            # dependencies such as 'numpy' could be added to the end of this command's list
            subprocess.call([str(py_exec),"-m", "pip", "install", "--target={}".format(py_prefix), package_name])
            
            # pip.main(['install', package_name, '--target', (sys.exec_prefix) + '/lib/site-packages'])
            
            importlib.import_module(import_name)
            print(f'Installation of {package_name} in Blender was a success !')
        
        except:
            print(f'Error : could not install {package_name} in Blender!', file=sys.stderr)