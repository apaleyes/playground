var a = String.raw`console.log("var a = " + b + a + c + "\nvar b = '" + b + "'\nvar c = '" + c  + "'\n\n" + a)`
var b = 'String.raw`'
var c = '`'

console.log("var a = " + b + a + c + "\nvar b = '" + b + "'\nvar c = '" + c  + "'\n\n" + a)