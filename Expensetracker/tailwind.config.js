// tailwind.config.js
module.exports = {
    content: [
      './node_modules/flyonui/dist/js/*.js',
    ],
    plugins: [
      require('flyonui'), 
      require('flyonui/plugin')
    ],
  }