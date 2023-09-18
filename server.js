const express = require('express');
const app = express();
var cors = require('cors')

app.use(cors())

app.listen(8080, function(){
    console.log('listening on 8080')

});

app.get('/pet', function(req, res){
    res.send('펫쇼핑 페이지');

});

app.get('/beauty-', function(req, res){
    res.send('뷰티')
});

app.get('/sound/:name', (req, res) => {
    const { name } = req.params

    console.log(name)
    if (name == "dog"){
        res.json({'sound': '멍멍'})
    }
    else if (name == "cat"){
        res.json({'sound': '냐옹'})
    }
})

app.get('/', function(req, res){
    res.sendFile(__dirname + "/index.html")
});
