line1 = "if __name__ == '__main__':"
line2 = r"    print('{1} = \"{0}\"\n{3} = r\"{2}\"\n\n{0}\n{2}'.format(line1, 'line1', line2, 'line2'))"

if __name__ == '__main__':
    print('{1} = \"{0}\"\n{3} = r\"{2}\"\n\n{0}\n{2}'.format(line1, 'line1', line2, 'line2'))