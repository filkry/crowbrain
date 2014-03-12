var page = require('webpage').create();
page.open('{{ url }}', function() {
    page.render('{{ output_file }}');
    phantom.exit();
});
