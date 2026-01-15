css = """
<style>
.chat-box {
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 10px;
}
.user {
    background-color: #2b313e;
}
.bot {
    background-color: #475063;
}
</style>
"""

user_template = """
<div class="chat-box user">
<b>You🚀:</b> {{MSG}}
</div>
"""

bot_template = """
<div class="chat-box bot">
<b>Answer🤖:</b> {{MSG}}
</div>
"""
