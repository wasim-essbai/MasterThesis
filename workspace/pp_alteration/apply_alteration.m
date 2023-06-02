function altered_signal = apply_alteration(signal, alteration_type, alt_level)
    function altered_signal = add_gaussian_noise(signal, sigma)
        altered_signal = signal + wgn(1,length(signal),sigma^2,'linear');
    end

    switch(alteration_type)
        case 'gwn'
            altered_signal = add_gaussian_noise(signal, alt_level);
    end
end
