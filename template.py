css = '''
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start; /* Align items to the top */
}

.chat-message.user {
    background-color: #F0F0F0;
    color: #333; /* Text color for bot messages */
}

.chat-message.bot {
    background-color: #F0F0F0;
    color: #333; /* Text color for bot messages */
}

.chat-message .avatar {
    width: 40px; /* Adjust the avatar size */
    height: 40px; /* Adjust the avatar size */
    margin-right: 0.5rem; /* Add spacing between avatar and message */
}

.chat-message .avatar img {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    flex: 1; /* Take up remaining space */
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #fff; /* Message bubble background color */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Add a subtle shadow to the message bubble */
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/128/11790/11790062.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/128/11772/11772273.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''