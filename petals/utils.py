def parse_ip_port(ip_port):
    ip, port =  ip_port.split(':') # ip
    port = int(port)
    return ip, port