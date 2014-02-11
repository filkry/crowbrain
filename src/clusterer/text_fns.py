def get_line_indent(l):
  depth_count = 0
  l = l.strip()
  front = l
  while front and front[0] == '-':
    depth_count = depth_count + 1
    front = front[1:]
  return depth_count

def extract_idea_ids_from_text(s):
  matches = []
  for match in re.findall('\(_id:[0-9]+\)', s):
    matches.append(int(re.findall('[0-9]+', match)[0]))
  return matches


