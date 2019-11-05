from flask import Flask, render_template, request
import smtplib
app = Flask(__name__)

@app.route('/success/', methods=['POST'])
def success():
    reciver = request.form['eaddress']
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()

    server.login('dandeproductions.contact@gmail.com', 'livelive127**')

    subject = "Access granted!"
    body = 'Hello, \n Access to your software has been granted.\n Download the file below \n\n \n\n https://drive.google.com/open?id=15FyijU2hGWjXn546KVT7A4aaKIRAyzIV'

    msg = f"Subject: {subject}\n\n{body}"

    server.sendmail(
            'dandeproductions.contact@gmail.com',
            reciver,
            msg
        )
    print('HEY EMAIL HAS BEEN SENT')

    server.quit()

    return render_template('success.html')

@app.route('/register/', methods=['GET'])
def register():
    return render_template('register.html')





@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
