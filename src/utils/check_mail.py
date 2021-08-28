import smtplib
from email.mime.text import MIMEText

def send_mail(senderAddr, recipientAddr, password, subject, text):

    # 세션 생성
    s = smtplib.SMTP('smtp.gmail.com', 587)

    # TLS 보안 시작
    s.starttls()

    # 로그인 인증

    s.login(senderAddr, password)

    msg=MIMEText(text)
    msg['Subject']=subject
    msg['From']=senderAddr
    msg['To']=recipientAddr

    s.sendmail(senderAddr, [recipientAddr], msg.as_string())
    s.quit()

if __name__ == '__main__':
    senderAddr = 'hwk0702@gmail.com'
    recipientAddr = 'hwk0702@gmail.com'
    password = 'fjwfqwjalwvnqagf'
    subject = 'test'
    text = 'test'
    send_mail(senderAddr, recipientAddr, password, subject, text)
