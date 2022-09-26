let model_select = document.getElementById('model');
let endpoint_selects = document.getElementsByName('endpoint');

function update_models(endpoint) {
  fetch('/model/' + endpoint).then(function(response) {
    response.json().then(function(data) {

      let optionHTML = '';
      for (let model of data.models) {
        optionHTML += '<option value="' + model.idx + '">' + model.name + '</option>';
      };

       model_select.innerHTML = optionHTML;
    });
  });
};

for (var i = endpoint_selects.length - 1; i >= 0; i--) {
  let endpoint_select = endpoint_selects[i]
  endpoint_select.onchange = function() {update_models(endpoint_select.value)};

  if (endpoint_select.checked) {
    update_models(endpoint_select.value);
  };
};