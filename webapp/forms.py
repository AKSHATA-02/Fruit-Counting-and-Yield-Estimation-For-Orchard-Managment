from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django import forms

class CreateUserForm(UserCreationForm):
	class Meta:
		model = User
		fields = ['username', 'email', 'password1', 'password2']

class LoginForm(AuthenticationForm):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		# Add Bootstrap form-control class to fields
		self.fields['username'].widget.attrs.update({
			'class': 'form-control'
		})
		self.fields['password'].widget.attrs.update({
			'class': 'form-control'
		})

