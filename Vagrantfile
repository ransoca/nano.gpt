# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "bento/ubuntu-22.04"

  config.vm.provision "file",
    source: "~/.ssh/id_ecdsa.pub",
    destination: "~/.ssh/authorized_keys"

  config.ssh.insert_key = false
  config.ssh.private_key_path = ["~/.ssh/id_ecdsa", "~/.vagrant.d/insecure_private_key"]

  config.vm.provision "ansible_local" do |ansible|
    ansible.playbook = ".ansible/main.yml"
    ansible.inventory_path = ".ansible/main.ini"
    ansible.limit = "all"
  end
end
