import sys
import docker

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def run(username, password):
    # docker
    client = docker.from_env()
    # login
    client.login(username=username, password=password, registry='https://registry-1.docker.io/')
    # load image
    img = client.images.pull('johbrob/rq-torch-env')
    # should maybe try to find available port
    # run container of image
    try:
        container = client.containers.run('johbrob/rq-torch-env:latest', detach=True, ports={'5000': 3000},
                                          name='webapp_torch_model_server')
    except docker.errors.APIError as e:
        stop_container = query_yes_no(
            "Container with name 'webapp_torch_model_server' is already running. Is it ok to stop and remove it?\n")
        if stop_container:
            container = client.containers.get('webapp_torch_model_server')
            container.stop()
            container.remove()
            container = client.containers.run('johbrob/rq-torch-env:latest', detach=True, ports={'5000': 3000},
                                              name='webapp_torch_model_server')
        else:
            print('terminating...')

def r():
    client = docker.from_env()
    container = client.containers.run('johbrob/rq-torch-env:latest', detach=True, ports={'5000': 3000},
                                      name='webapp_torch_model_server')
    return container

def main():
    import os
    print(f'beginning running subprocess {os.getpid()}\n')
    #run(sys.argv[1], sys.argv[2])
    a = r()
    print(f'finishing running subprocess {os.getpid()}\n')
    #return [sys.argv[0], sys.argv[1]]
    return a


if __name__ == '__main__':
    main()

