from django.shortcuts import render, redirect, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from userincome.models import Source
from .forms import User_Profile, FamilyMemberProfileForm, DemographicProfileForm
from .models import Profile

@login_required(login_url='/authentication/login')
def userprofile(request):
    try:
        profile = request.user.profile
    except Profile.DoesNotExist:
        profile = Profile.objects.create(user=request.user, account_type='SOLO', profile_type='OWNER')
    
    Sources = Source.objects.filter(owner=request.user)
    family_members = []
    
    if profile.account_type == 'MULTI' and profile.is_owner():
        family_members = Profile.objects.filter(owner=request.user)
    
    if request.method == "POST":
        if 'update_profile' in request.POST:
            form = User_Profile(data=request.POST, instance=request.user)
            demographic_form = DemographicProfileForm(data=request.POST, instance=profile)
            
            if form.is_valid() and demographic_form.is_valid():
                form.save()
                demographic_form.save()
                messages.success(request, 'Profile Updated Successfully!!')
            else:
                # If either form is invalid, prepare error messages
                for field, errors in form.errors.items():
                    for error in errors:
                        messages.error(request, f"Error in {field}: {error}")
                for field, errors in demographic_form.errors.items():
                    for error in errors:
                        messages.error(request, f"Error in {field}: {error}")
        else:
            form = User_Profile(instance=request.user)
            demographic_form = DemographicProfileForm(instance=profile)
    else:
        form = User_Profile(instance=request.user)
        demographic_form = DemographicProfileForm(instance=profile)
    
    context = {
        'form': form,
        'demographic_form': demographic_form,
        'sources': Sources,
        'profile': profile,
        'family_members': family_members,
        'family_member_form': FamilyMemberProfileForm()
    }
    return render(request, 'userprofile/profile.html', context)

@login_required(login_url='/authentication/login')
def update_demographics(request):
    if request.method == "POST":
        try:
            profile = request.user.profile
            form = DemographicProfileForm(data=request.POST, instance=profile)
            
            if form.is_valid():
                form.save()
                messages.success(request, 'Demographic information updated successfully!')
            else:
                for field, errors in form.errors.items():
                    for error in errors:
                        messages.error(request, f"Error in {field}: {error}")
        except Profile.DoesNotExist:
            profile = Profile.objects.create(user=request.user, account_type='SOLO', profile_type='OWNER')
            form = DemographicProfileForm(data=request.POST, instance=profile)
            
            if form.is_valid():
                form.save()
                messages.success(request, 'Demographic information saved successfully!')
            else:
                for field, errors in form.errors.items():
                    for error in errors:
                        messages.error(request, f"Error in {field}: {error}")
    
    return redirect('account')

@login_required(login_url='/authentication/login')
def add_family_member(request):
    if request.method != "POST":
        return redirect('account')
        
    try:
        owner_profile = request.user.profile
        if not owner_profile.is_owner() or owner_profile.account_type != 'MULTI':
            messages.error(request, "You don't have permission to add family members")
            return redirect('account')
        
        form = FamilyMemberProfileForm(request.POST)
        if form.is_valid():
            username = request.POST.get('username')
            email = request.POST.get('email')
            relationship = form.cleaned_data['relationship']
            
            # Create user account for family member
            user = User.objects.create_user(username=username, email=email)
            user.set_password(request.POST.get('password'))
            user.save()
            
            # Create profile for family member
            Profile.objects.create(
                user=user,
                profile_type='MEMBER',
                account_type='MULTI',
                owner=request.user,
                relationship=relationship
            )
            
            messages.success(request, f'Family member {username} added successfully')
        else:
            messages.error(request, 'Invalid form submission')
    except Exception as e:
        messages.error(request, f'Error adding family member: {str(e)}')
    
    return redirect('account')

@login_required(login_url='/authentication/login')
def remove_family_member(request, member_id):
    try:
        owner_profile = request.user.profile
        if not owner_profile.is_owner() or owner_profile.account_type != 'MULTI':
            messages.error(request, "You don't have permission to remove family members")
            return redirect('account')
            
        member_profile = Profile.objects.get(id=member_id, owner=request.user)
        user = member_profile.user
        member_profile.delete()
        user.delete()
        messages.success(request, "Family member removed successfully")
    except Profile.DoesNotExist:
        messages.error(request, "Family member not found")
    except Exception as e:
        messages.error(request, f"Error removing family member: {str(e)}")
    
    return redirect('account')

def addSource(request):
    if request.method == "POST":
        newSource = request.POST['Source']
        if Source.objects.filter(name=newSource, owner=request.user).exists():
            messages.warning(request, "Income source already Exists")
            return HttpResponseRedirect('/account/')
        if len(newSource) == 0:
            return HttpResponseRedirect('/account/')
        newsourceadded = Source.objects.create(name=newSource, owner=request.user)
        newsourceadded.save()
        messages.success(request, 'Source added successfully')
    return HttpResponseRedirect('/account/')