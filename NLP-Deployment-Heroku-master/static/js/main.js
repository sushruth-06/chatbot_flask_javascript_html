
$(document).ready(function () {

    

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-message')[0]);
        var mes = document.getElementById('message').value
        

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                
                document.getElementById("name").innerHTML += "<li>" + mes + "</li>";
                document.getElementById("name").innerHTML += "<li>" + data + "</li>";
                console.log('Success!'+ inputs+outputs);
            },
        });
    });

});
