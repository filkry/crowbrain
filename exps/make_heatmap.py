#!/usr/bin/python

import json

html_template = '''
<html>
  <title>Brainstorming Heatmaps</title>
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>
  <script type="text/javascript">
    var table_data = %s;
    function make_table(table_data) {
      var return_str = '<table id="data_table">';
      var headers = table_data["headers"];
      var checked_headers = [];
      return_str += "<tr>";
      for (var i = 0; i < headers.length; i++) {
        var header_name = headers[i];
        console.log(header_name + " Checked: " + $("#"+header_name).prop('checked'));
        if ($("#" + header_name).prop('checked')) {
          console.log("Writing " + header_name);
          return_str += "<td>" + header_name + "</td>";
          checked_headers.push(header_name);
        }
      }
      return_str += "</tr>";
      for (var i = 0; i < table_data["rows"].length; i++) {
        return_str += "<tr>";
        var row = table_data["rows"][i];
        for (var j = 0; j < checked_headers.length; j++) {
          var table_cell_color = row[checked_headers[j]];
          console.log("Color: " + table_cell_color);
          return_str += '<td bgcolor="' + table_cell_color + '">&nbsp;</td>';
        }
        return_str += "</tr>";
      }
      return_str += "</table>";
      return return_str;
    }
    function make_checkboxes(table_data) {
      var return_str = "<form>";
      var headers = table_data['headers'];
      for (var i = 0; i < headers.length; i++) {
        var header_name = headers[i];
        return_str += '<input type="checkbox" id="' + header_name + '" name="' + header_name + '" value="' + header_name + '" checked="1">' + header_name + '<br>';
      }
      return_str += "</form>";
      return return_str;
    }
    function add_listeners(table_data) {
      var headers = table_data['headers'];
      for (var i = 0; i < headers.length; i++) {
        var header_name = headers[i];
        $("#" + header_name).change(function() {
          $("#data_table").replaceWith(make_table(table_data));
        });
      }
    }
    $(document).ready(function() {
    $("#checkboxes").append(make_checkboxes(table_data));
    $("#data_table").replaceWith(make_table(table_data));
    add_listeners(table_data);
    });
  </script>
  <body>
    <div id="checkboxes">
      Show:
    </div>
    <div id="tables">
      <table id="data_table">
      </table>
    </div>
  </body>
</html>
'''

def process_data(data_list):
  for data_dict in data_list:
    for key in data_dict:
      val = data_dict[key]
      if type(val) == int:
        if val:
          data_dict[key] = "#000000"
        else:
          data_dict[key] = "#FFFFFF"
      else:
        if val > 1.0:
          print "Error: Value is >1", val
          val = 1.0
        if val < 0:
          print "Error: Value is <0", val
          val = 0
        as_int = int((1.0-val)*255)
        data_dict[key] = "#%02x%02x%02x" % (as_int, as_int, as_int)

def make_data_dict(data_list):
  headers = []
  for row_dict in data_list:
    for key in row_dict:
      if not key in headers:
        headers.append(key)
  data_dict = { 'headers' : headers, 'rows' : data_list }
  return data_dict

def output_heatmap(data_list, output_fname):
  process_data(data_list)
  data_dict = make_data_dict(data_list)
  html = html_template % json.dumps(data_dict)
  with open(output_fname, 'w') as fout:
    print "Writing", output_fname
    fout.write(html)

if __name__ == "__main__":
  test_data = [
    { "A" : 0, "B" : 0.25, "C" : 0.75},
    { "A" : 1, "B" : 0.55, "C" : 0.25},
    { "A" : 0, "B" : 0.75, "C" : 0.55}
  ]
  output_heatmap(test_data, "heatmap.html")
