#!/usr/bin/python

import json, math

html_template = '''
<html>
  <title>Brainstorming Heatmaps</title>
  <style type='text/css'>
  </style>
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>
  <script type="text/javascript">
    var table_datas = %s;

    function make_tables(table_datas) {
      var return_str = '<table><tr>';
      for (var table_key in table_datas['tables']) {
        table_data = table_datas['tables'][table_key];
        return_str += "<td style='vertical-align:top; border-right:1px solid black; padding-left:5px; padding-right:5px;'><div id='data_table_" + table_key + "'>";
        return_str += make_table(table_data, table_datas['headers']);
        return_str += "</div></td>";
      }
      return return_str
    }

    function make_table(table_data, headers) {
      var return_str = '<table id="data_table">';
      return_str += "<tr>";
      for (var i = 0; i < headers.length; i++) {
        var header_name = headers[i];
        return_str += "<td class='" + header_name + "'>" + header_name + "</td>";
      }
      return_str += "</tr>";
      for (var i = 0; i < table_data.length; i++) {
        return_str += "<tr>";
        var row = table_data[i];
        for (var j = 0; j < headers.length; j++) {
          var table_cell_color = row[headers[j]];
          //console.log("Color: " + table_cell_color);
          return_str += '<td bgcolor="' + table_cell_color + '" class="' + headers[j] + '">&nbsp;</td>';
        }
        return_str += "</tr>";
      }
      return_str += "</table>";
      return return_str;
    }
    function make_checkboxes(table_datas) {
      var return_str = "<form>";
      var headers = table_datas['headers'];
      for (var i = 0; i < headers.length; i++) {
        var header_name = headers[i];
        return_str += '<input type="checkbox" id="' + header_name + '" name="' + header_name + '" value="' + header_name + '" checked="1">' + header_name + '<br>';
      }
      return_str += "</form>";
      return return_str;
    }
    function add_listeners(table_datas) {
      var headers = table_datas['headers'];
      for (var i = 0; i < headers.length; i++) {
        var header_name = headers[i];
        $("#" + header_name).change(function() {
          id = $(this).attr("id")
          if($("#" + id).prop('checked')) {
            $("." + id).css("display", "block");
          } else {
            $("." + id).css("display", "none");
          }
        });
      }
    }
    $(document).ready(function() {
      $("#checkboxes").append(make_checkboxes(table_datas));
      $("#data_tables").replaceWith(make_tables(table_datas));
      add_listeners(table_datas);
    });
  </script>
  <body>
    <div id="checkboxes">
      Show:
    </div>
    <div id="tables">
      <table id="data_tables">
      </table>
    </div>
  </body>
</html>
'''

def process_data(data_lists):
  for key in data_lists:
    data_list = data_lists[key]
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
          elif val < 0:
            print "Error: Value is <0", val
            val = 0
          if math.isnan(val):
            #print "Error, value is None"
            val = 0
          as_int = int((1.0-val)*255)
          data_dict[key] = "#%02x%02x%02x" % (as_int, as_int, as_int)

def make_data_dict(data_lists):
  headers = []
  for row_dict in data_lists[data_lists.keys()[0]]:
    for key in row_dict:
      if not key in headers:
        headers.append(key)
  data_dict = { 'headers' : headers, 'tables' : data_lists }
  return data_dict

def output_heatmap(data_lists, output_fname):
  process_data(data_lists)
  data_dict = make_data_dict(data_lists)
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
